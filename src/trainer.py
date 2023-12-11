import numpy as np
import torch
import json
import warnings
# from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from custom_trainer import Trainer, get_scheduler
from src.loss_and_metrics import get_ae_loss, CWEncoderLoss, AllMetrics, pykeops_chamfer, cpu_chamfer
from src.dataset import get_datasets
from src.model import get_model
# match cost is the emd version used in other works as a baseline
from structural_losses import match_cost


class AETrainer(Trainer):
    saved_accuracies = {}

    def __init__(self, model, **block_args):
        super().__init__(model, **block_args)
        self.metrics = self.loss
        return

    def test(self, partition='val', save_outputs=False, all_metrics=False, de_normalize=False, **_):
        if all_metrics:
            self.metrics = lambda x, y, z: AllMetrics(de_normalize)(x, y)
        super().test(partition=partition, save_outputs=save_outputs)
        return

    def update_m_training(self, m):
        self.model.m_training = m

    def loss_dict(self, outputs, inputs, targets, **_):
        return self.loss(outputs, inputs, targets)

    def helper_inputs(self, inputs):
        # inputs length vary on the dataset, when resampling two different re-samplings of the shape are given
        indices = inputs[-1]
        if torch.all(indices == 0):
            indices = None
        input_shape = inputs[1]
        return [input_shape], {'indices': indices}

    def metric_dict(self, outputs, inputs, targets, **_):
        return self.metrics(outputs, inputs, targets)


class VQVAETrainer(AETrainer):

    def test(self, partition='val', save_outputs=True, all_metrics=False, de_normalize=False, **kwargs):
        super().test(partition, save_outputs, all_metrics, de_normalize, **kwargs)
        idx = torch.stack(self.test_outputs['one_hot_idx']).float()
        idx = idx.mean(0)  # average dataset values
        self.print_statistics('Index Usage', idx)
        print(torch.histc(idx))
        return

    # def gmm_sampling(self):
    #     with self.model.double_encoding:
    #         self.test(partition='train', save_outputs=True)
    #     z = torch.stack(self.test_outputs['z']).detach().numpy()
    #     self.model.gm = GaussianMixture(32).fit(z)
    #     return

    def evaluate_generated_set(self, partition, repeat_chamfer=10, repeat_emd=0, batch_size=32, oracle=False):
        # self.gmm_sampling()
        loader = self.test_loader if partition == 'test' else self.val_loader
        test_dataset = []
        for batch_idx, (inputs, targets, index) in enumerate(loader):
            test_clouds = inputs[1]
            test_dataset.extend(test_clouds.cpu())
        test_l = len(test_dataset)
        mmd_chamfer = []
        mmd_emd = []
        cov_chamfer = []
        cov_emd = []
        nna_chamfer = []
        nna_emd = []
        for test_idx in range(max(repeat_chamfer, repeat_emd)):
            print(f'Test {test_idx + 1}:')
            # Generating the samples
            generated_dataset = []
            if oracle:
                train_ds = self.train_loader.dataset
                sample_train = np.random.choice(range(len(train_ds)), size=test_l, replace=True)
                generated_dataset = [train_ds[i][0][1] for i in sample_train]
                print('Training Dataset has been sampled.')
            else:
                while len(generated_dataset) < test_l:
                    batch = min(batch_size, test_l - len(generated_dataset))
                    samples = self.model.random_sampling(batch)['recon'].detach().cpu()
                    samples -= samples.mean(dim=1, keepdim=True)
                    std = torch.max(torch.sqrt(torch.sum(samples ** 2, dim=2)), dim=1)[0].view(-1, 1, 1)
                    samples /= std
                    generated_dataset.extend(samples)
                print('Random Dataset has been generated.')
            assert len(test_dataset) == len(generated_dataset)
            all_shapes = test_dataset + generated_dataset
            if repeat_chamfer:
                mmd, coverage, nna = self.gen_metrics(all_shapes, 'Chamfer', batch_size, self.device)
                mmd_chamfer.append(mmd)
                cov_chamfer.append(coverage)
                nna_chamfer.append(nna)
                repeat_chamfer -= 1

            if repeat_emd:
                mmd, coverage, nna = self.gen_metrics(all_shapes, 'Emd', batch_size, self.device)
                mmd_emd.append(mmd)
                cov_emd.append(coverage)
                nna_emd.append(nna)
                repeat_emd -= 1

        self.print_statistics('Minimum Matching Distance score (Chamfer): ', mmd_chamfer)
        self.print_statistics('Coverage score (Chamfer): ', cov_chamfer)
        self.print_statistics('1-NNA score (Chamfer): ', nna_chamfer)
        self.print_statistics('Minimum Matching Distance score (Emd): ', mmd_emd)
        self.print_statistics('Coverage score (Emd): ', cov_emd)
        self.print_statistics('1-NNA score (Emd): ', nna_emd)
        return

    @staticmethod
    def gen_metrics(all_shapes, metric, batch_size, device):
        l: int = len(all_shapes)
        test_l = l // 2
        dist_array = np.zeros((l, l), dtype=float)
        for i, cloud in enumerate(all_shapes):
            dist_array[i, i] = np.inf  # safe way of ignoring this entry (ignored by MMD, COV and NNA)
            for j in range(i + 1, len(all_shapes), batch_size):
                clouds1 = torch.stack(all_shapes[j:j + batch_size]).to(device)
                clouds2 = cloud.unsqueeze(0).expand_as(clouds1).to(device)
                if metric == 'Chamfer':
                    chamfer = pykeops_chamfer if device.type == 'cuda' else cpu_chamfer
                    dist = chamfer(clouds1, clouds2) / cloud.shape[0]
                else:
                    dist = match_cost(clouds1.contiguous(), clouds2.contiguous()) / cloud.shape[0]
                dist_array[i, j:j + len(clouds1)] = dist.cpu()
                dist_array[j:j + len(clouds1), i] = dist.cpu()
        closest = dist_array.argmin(axis=1)  # np.inf on the diagonal so argmin != index
        test_closest = closest[:test_l]
        generated_closest = closest[test_l:]
        nna = ((test_closest < test_l).sum() + (generated_closest >= test_l).sum()) / l
        coverage_array = dist_array[test_l:, :test_l]  # gen_sample, ref_sample
        coverage_closest = coverage_array.argmin(axis=1)
        coverage = np.unique(coverage_closest).shape[0] / test_l
        mmd_array = dist_array[:test_l, test_l:]  # ref_sample, gen_sample
        mmd = mmd_array.min(axis=1).mean()
        print(f'Minimum Matching Distance score ({metric}): {mmd:.4e}')
        print(f'Coverage score ({metric}): {coverage:.4e}')
        print(f'1-NNA score ({metric}): {nna:.4e}')
        return mmd, coverage, nna

    @staticmethod
    def print_statistics(name, test_outcomes):
        if not len(test_outcomes):
            print(f'No test performed for "{name}"')
            return
        test_outcomes = np.array(test_outcomes)
        print(name)
        print(f'Number of tests: {len(test_outcomes):} Min: {test_outcomes.min():.4e} Max: {test_outcomes.max():.4e} '
              f'Mean: {test_outcomes.mean():.4e} Std: {test_outcomes.std():.4e}')


class CWTrainer(Trainer):

    def __init__(self, vqvae_trainer, **block_args):
        self.vqvae_trainer = vqvae_trainer
        self.vqvae_model = vqvae_trainer.model
        self.vqvae_epoch = vqvae_trainer.epoch
        super().__init__(vqvae_trainer.model.cw_encoder, **block_args)
        return

    def test(self, partition='val', save_outputs=False, all_metrics=False, de_normalize=False, **_):
        super().test(partition=partition, save_outputs=save_outputs)  # test partition uses val dataset
        with self.vqvae_trainer.model.double_encoding:
            self.vqvae_trainer.test(partition, all_metrics, de_normalize, save_outputs)  # test on val

    def loss_dict(self, output, inputs, targets, **_):
        return self.loss(output, inputs, targets)

    def save(self, new_exp_name=None):
        self.model.eval()
        paths = self.vqvae_trainer.paths(new_exp_name or self.exp_name, self.vqvae_epoch, self.model_pardir)
        self.vqvae_model.cw_encoder.load_state_dict(self.model.state_dict())
        if new_exp_name:
            json.dump(self.settings, open(paths['settings'], 'w'))
        torch.save(self.vqvae_model.state_dict(), paths['model'])
        print('Model saved at: ', paths['model'])
        return

    @torch.inference_mode()
    def helper_inputs(self, inputs):
        [_, *clouds, _] = inputs
        self.vqvae_model.train(self.model.training)
        cw_q = self.vqvae_model.encoder(clouds[0], None)
        _, one_hot_idx = self.vqvae_model.quantise(cw_q)
        return [cw_q.detach().clone()], {'data': {'one_hot_idx': one_hot_idx.detach().clone()}}

    def show_latent(self):
        if not self.check_visdom_connection():
            warnings.warn('Impossible to show latent space on visdom. Check the connection.')
            return
        test_mu = torch.stack(self.vqvae_trainer.test_outputs['mu'])
        pca = PCA(3)
        test_mu_pca = pca.fit_transform(test_mu.numpy())
        test_labels = np.ones(test_mu_pca.shape[0])
        pseudo_mu = self.model.pseudo_mu
        pseudo_mu_pca = pca.transform(pseudo_mu.detach().cpu().numpy())
        pseudo_labels = 2 * np.ones(pseudo_mu_pca.shape[0])
        mu_pca = np.vstack((test_mu_pca, pseudo_mu_pca))
        labels = np.hstack((test_labels, pseudo_labels))
        title = 'Continuous Latent Space'
        self.vis.scatter(X=mu_pca, Y=labels, win=title,
                         opts=dict(title=title, markersize=5, legend=['Validation', 'Pseudo-Inputs']))


def get_trainer(args):
    model = get_model(**vars(args))
    train_dataset, val_dataset, test_dataset = get_datasets(**vars(args))
    if args.model_head == 'VQVAE':
        lr = {'encoder': args.lr, 'decoder': args.lr, 'cw_encoder': args.lr}
    else:
        lr = args.lr
    scheduler_cls = get_scheduler(args.scheduler_name)
    if args.scheduler_name == 'Cosine':
        scheduler = scheduler_cls(decay_steps=args.decay_steps, min_decay=args.min_decay)
    else:
        scheduler = scheduler_cls()
    trainer_args = dict(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        optim_args={'lr': lr, 'weight_decay': args.wd},
        scheduler=scheduler,
        loss=get_ae_loss(**vars(args)),
        **vars(args)
    )

    return (VQVAETrainer if args.model_head == 'VQVAE' else AETrainer)(model, **trainer_args)


def get_cw_trainer(vqvae_trainer,  args):
    train_dataset, val_dataset, test_dataset = get_datasets(**vars(args))
    del args.batch_size  # clashes with the optimizer_cls argument
    del args.optimizer_cls  # clashes with the optimizer_cls argument
    scheduler_cls = get_scheduler(args.vae_scheduler_name)
    if args.scheduler_name == 'Cosine':
        scheduler = scheduler_cls(decay_steps=args.vae_decay_steps, min_decay=args.vae_min_decay)
    else:
        scheduler = scheduler_cls()
    trainer_args = dict(
        batch_size=args.vae_batch_size,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        optimizer_cls=args.vae_optimizer_cls,
        optim_args={'lr': args.vae_lr, 'weight_decay': args.vae_wd},
        scheduler=scheduler,
        loss=CWEncoderLoss(**vars(args)),
        **vars(args)
    )
    return CWTrainer(vqvae_trainer, **trainer_args)
