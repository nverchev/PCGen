import os
import sys
import numpy as np
import torch
import json
from sklearn import svm, metrics
from sklearn.decomposition import PCA
from src.trainer_base import Trainer
from src.optim import get_opt, CosineSchedule
from src.loss_and_metrics import get_ae_loss, CWEncoderLoss, AllMetrics
from src.viz_pc import show_pc
from src.neighbour_op import square_distance
from src.loss_and_metrics import chamfer

# match cost is the emd version used in other works as a baseline
from structural_losses import match_cost


class AETrainer(Trainer):
    clf = svm.SVC(kernel='linear')
    saved_accuracies = {}

    def __init__(self, model, block_args):
        super().__init__(model, **block_args)
        self.acc = None
        self.cf = None  # confusion matrix
        self._loss = get_ae_loss(**block_args)
        self._metrics = self._loss
        return

    def test(self, partition, all_metrics=False, de_normalize=False, save_outputs=False, **kwargs):
        if all_metrics:
            self._metrics = lambda x, y, z: AllMetrics(de_normalize)(x, y)
        super().test(partition=partition, save_outputs=save_outputs)
        return

    def update_m_training(self, m):
        self.model.m_training = m

    def class_metric(self, final=False):
        # No rotation here
        self.train_loader.dataset.rotation = False
        self.test(partition='train', save_outputs=self.test_outputs)
        self.train_loader.dataset.rotation = True
        x_train = np.array([cw.numpy() for cw in self.test_outputs['cw']])
        y_train = np.array([cw.numpy() for cw in self.test_metadata['test_targets']])
        shuffle = np.random.permutation(y_train.shape[0])
        x_train = x_train[shuffle]
        y_train = y_train[shuffle]
        print('Fitting the classifier ...')
        self.clf.fit(x_train, y_train)
        partition = 'test' if final else 'val'
        self.test(partition=partition)
        x_test = np.array([cw.numpy() for cw in self.test_outputs['cw']])
        y_test = np.array([cw.numpy() for cw in self.test_metadata['test_targets']])
        y_hat = self.clf.predict(x_test)
        self.acc = (y_hat == y_test).sum() / y_hat.shape[0]
        print('Accuracy: ', self.acc)
        self.cf = metrics.confusion_matrix(y_hat, y_test, normalize='true')
        print('Mean Accuracy;', np.diag(self.cf).astype(float).mean())
        directory = os.path.join(self.model_pardir, self.exp_name)
        accuracy_path = os.path.join(directory, 'svm_accuracies.json')
        self.saved_accuracies[self.epoch] = self.acc
        json.dump(self.saved_accuracies, open(accuracy_path, 'w'))
        return self.acc

    def latent_visualisation(self, highlight_label):
        from sklearn.decomposition import PCA
        cw = torch.stack(self.test_outputs['cw'])
        pca = PCA(3)
        cw_pca = pca.fit_transform(cw.numpy())
        labels = torch.stack(self.test_metadata['test_targets']).cpu().numpy()
        highlight_cw = cw_pca[(highlight_label == labels)]
        show_pc([torch.FloatTensor(cw_pca), highlight_cw], colors=['blue', 'red'])

    def loss(self, output, inputs, targets):
        return self._loss(output, inputs, targets)

    def helper_inputs(self, inputs, labels):
        # inputs length vary on the dataset, when resampling two different re-samplings of the shape are given
        indices = inputs[-1]
        if torch.all(indices == 0):
            indices = None
        input_shape = inputs[1]
        return {'x': input_shape, 'indices': indices}

    def metrics(self, output, inputs, targets):
        return self._metrics(output, inputs, targets)


class VQVAETrainer(AETrainer):

    def test(self, partition, all_metrics=False, de_normalize=False, save_outputs=True, **kwargs):
        super().test(partition, all_metrics, de_normalize, save_outputs=True, **kwargs)
        idx = torch.stack(self.test_outputs['one_hot_idx']).float()
        idx = idx.mean(0)  # average dataset values
        self.print_statistics('Index Usage', idx)
        print(torch.histc(idx))
        return

    def evaluate_generated_set(self, partition, repeat_chamfer=10, repeat_emd=0, batch_size=32, oracle=False):
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
        for _ in range(max(repeat_chamfer, repeat_emd)):
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
                    pairwise_dist = square_distance(clouds1, clouds2)
                    dist = chamfer(clouds1, clouds2, pairwise_dist)[0] / cloud.shape[0]
                else:
                    dist = match_cost(clouds1.continuous(), clouds2.continuous()) / cloud.shape[0]
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
        print(f'Coverage score ({metric}): {coverage:.4e}')
        print(f'Minimum Matching Distance score ({metric}): {mmd:.4e}')
        print(f'1-NNA score ({metric}): {nna:.4e}')
        return mmd, coverage, nna

    @staticmethod
    def print_statistics(name, test_outcomes):
        if not len(test_outcomes):
            print(f'No test performed for "{name}"')
            return
        to = np.array(test_outcomes)
        print(name)
        print(f'Number of tests: {len(to):} Min: {to.min(initial=None):.4e} Max: {to.max(initial=None):.4e} '
              f'Mean: {to.mean():.4e} Std: {to.std():.4e}')


class CWTrainer(Trainer):

    def __init__(self, vqvae_trainer, block_args):
        self.vqvae_trainer = vqvae_trainer
        self.vqvae_model = vqvae_trainer.model
        self.vqvae_epoch = vqvae_trainer.epoch
        super().__init__(vqvae_trainer.model.cw_encoder, **block_args)
        self._loss = CWEncoderLoss(**block_args)
        return

    def test(self, partition, all_metrics=False, de_normalize=False, save_outputs=0, **kwargs):
        super().test(partition=partition, save_outputs=save_outputs)  # test partition uses val dataset
        with self.vqvae_trainer.model.double_encoding:
            self.vqvae_trainer.test(partition, all_metrics, de_normalize, save_outputs, **kwargs)  # test on val

    def loss(self, output, inputs, targets):
        return self._loss(output, inputs, targets)

    def save(self, new_exp_name=None):
        self.model.eval()
        paths = self.vqvae_trainer.paths(new_exp_name, epoch=self.vqvae_epoch)
        self.vqvae_model.cw_encoder.load_state_dict(self.model.state_dict())
        if new_exp_name:
            json.dump(self.settings, open(paths['settings'], 'w'))
        torch.save(self.vqvae_model.state_dict(), paths['model'])
        print('Model saved at: ', paths['model'])
        return

    def helper_inputs(self, inputs, labels):
        return {'x': inputs[0]}

    def show_latent(self):
        mu = torch.stack(self.vqvae_trainer.test_outputs['mu'])
        pca = PCA(3)
        cw_pca = pca.fit_transform(mu.numpy())
        pseudo_mu = self.model.pseudo_mu
        pseudo_mu_pca = pca.transform(pseudo_mu.detach().cpu().numpy())
        show_pc([torch.FloatTensor(cw_pca), torch.FloatTensor(pseudo_mu_pca)], colors=['blue', 'red'])


def get_trainer(model, loaders, args):
    if args.model_head == 'VQVAE':
        lr = {'encoder': args.lr, 'decoder': args.lr, 'cw_encoder': 1 * args.lr}
    else:
        lr = args.lr
    optimizer, optim_args = get_opt(args.opt_name, lr, args.wd)
    trainer_args = dict(
        optimizer=optimizer,
        optim_args=optim_args,
        scheduler=CosineSchedule(decay_steps=args.decay_steps, min_decay=args.min_decay),
        **loaders,
        **vars(args)

    )

    return (VQVAETrainer if args.model_head == 'VQVAE' else AETrainer)(model, trainer_args)


def get_cw_trainer(vqvae_trainer, cw_loaders, args):
    optimizer, optim_args = get_opt(args.vae_opt_name, args.vae_lr, args.vae_wd)
    trainer_args = dict(
        optimizer=optimizer,
        optim_args=optim_args,
        **cw_loaders,
        **vars(args)
    )
    return CWTrainer(vqvae_trainer, trainer_args)
