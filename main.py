import os
import torch
import numpy as np
import argparse
import pykeops
from src.dataset import get_loaders, get_cw_loaders
from src.optim import get_opt, CosineSchedule
from src.trainer import get_ae_trainer, CWTrainer
from src.model import get_model

pykeops.set_verbose(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud Encoder - Generator')

    parser.add_argument('--encoder', type=str, default='LDGCNN', choices=['LDGCNN', 'DGCNN', 'FoldingNet'])
    parser.add_argument('--decoder', type=str, default='PCGen',
                        choices=['PCGen', 'PCGenC', 'Full', 'FoldingNet', 'TearingNet',
                                 'AtlasNetDeformation', 'AtlasNetStructures'])
    parser.add_argument('--exp', type=str, default='',
                        help='Name of the experiment. If it starts with "final" the test set is used for eval.')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Default is given by "_".join(model(GF), recon_loss, exp)')
    parser.add_argument('--components', type=int, default=0,
                        help='Components of PCGenC or patches in AtlasNet, 0 is for default (resp. 8 and 10)')
    parser.add_argument('--gf', action='store_true', default=False, help='Graph filtering after decoder')
    parser.add_argument('--recon_loss', type=str, default='Chamfer',
                        choices=['Chamfer', 'ChamferA', 'ChamferS', 'Sinkhorn'], help='PC reconstruction loss')
    parser.add_argument('--ae', type=str, default='AE', choices=['Oracle', 'AE', 'VAE', 'VQVAE'],
                        help='Oracle is identity (measures resampling error), VQVAE adds quantisation')
    parser.add_argument('--dir_path', type=str, default='./', help='Directory for storing data and models')
    parser.add_argument('--dataset', type=str, default='Modelnet40',
                        choices=['Modelnet40', 'ShapenetAtlas', 'Coins', 'Faust', 'ShapenetFlow'])
    parser.add_argument('--select_classes', type=str, default=[], nargs='+',
                        help='selects only specific classes of the dataset')
    parser.add_argument('--num_points', type=int, default=0,
                        help='Number of (maximum) points of the training dataset, 0 is default for dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optim', type=str, default='AdamW', choices=['SGD', 'SGD_momentum', 'Adam', 'AdamW'],
                        help='SGD_momentum has momentum = 0.9')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate, 10x when for the second encoder')
    parser.add_argument('--wd', type=float, default=0.000001, help='Weight decay')
    parser.add_argument('--min_decay', type=float, default=0.01, help='fraction of the initial lr at the end of train')
    parser.add_argument('--k', type=int, default=20,
                        help='Number of neighbours of a point (counting the point itself) in DGCNN]')
    parser.add_argument('--cw_dim', type=int, default=1024, help='Dimension of the codeword space')
    parser.add_argument('--z_dim', type=int, default=64, help='Dimension of the second encoding space')
    parser.add_argument('--book_size', type=int, default=32, help='Dictionary size for vector quantisation')
    parser.add_argument('--dim_embedding', type=int, default=4, help='Dimension of the vector for vector quantisation')
    parser.add_argument('--c_reg', type=float, default=1, help='Coefficient for regularization')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Runs on CPU')
    parser.add_argument('--epochs', type=int, default=350, help='Number of total training epochs')
    parser.add_argument('--decay_period', type=int, default=250, help='Number of epochs before lr decays stops')
    parser.add_argument('--checkpoint', type=int, default=10, help='Number of epochs between checkpoints')
    parser.add_argument('--m_training', type=int, default=0,
                        help='Points generated when training,'
                             '-1 for  increasing sequence 128 -> 4096, 0 input number of points ')
    parser.add_argument('--m_test', type=int, default=0, help='Points generated when testing,'
                                                              ' 0 for input number of points')
    parser.add_argument('--ind', type=int, default=[0], nargs='+', help='index for reconstruction to visualize')
    parser.add_argument('--load', type=int, default=-1,
                        help='Load a saved model with the same settings. -1 for starting from scratch,'
                             '0 for most recent, otherwise epoch after which the model was saved')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Evaluate the model (exp_name needs to start with "final")')
    parser.add_argument('--minio_credential', type=str, default='',
                        help='Path of txt file with written server;access_key;secret_key')
    return parser.parse_args()


def main(task='train/eval'):
    assert task in ['train/eval', 'return model for profiling', 'return loaded model for random generation',
                    'train cw encoder', 'visualise reconstructions', 'evaluate random generation']
    args = parse_args()
    encoder_name = args.encoder
    decoder_name = args.decoder
    graph_filtering = args.gf
    experiment = args.exp
    recon_loss = args.recon_loss
    ae = args.ae
    components = args.components
    gf = 'GF' if graph_filtering else ''
    exp_name = args.model_path or '_'.join([encoder_name, decoder_name + gf, recon_loss, ae, experiment])
    final = experiment[:5] == 'final'
    dir_path = args.dir_path
    dataset_name = args.dataset
    select_classes = args.select_classes
    if args.num_points:
        num_points = args.num_points
    else:
        if dataset_name in ['Modelnet40', 'ShapenetFlow']:
            num_points = 2048
        elif dataset_name in ['Faust']:
            num_points = 6890
        else:
            num_points = 2500
    batch_size = args.batch_size
    k = args.k
    opt_name = args.optim
    initial_learning_rate = args.lr
    weight_decay = args.wd
    cw_dim = args.cw_dim
    z_dim = args.z_dim
    book_size = args.book_size
    dim_embedding = args.dim_embedding
    c_reg = args.c_reg
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    training_epochs = args.epochs
    decay_period = args.decay_period
    min_decay = args.min_decay
    checkpoint_every = args.checkpoint
    m = args.m_test if args.m_test else num_points
    denormalise = dataset_name in ['ShapenetFlow']
    m_training = args.m_training if args.m_training else num_points
    ind = args.ind
    load = args.load
    model_eval = args.eval
    minio_credential = args.minio_credential

    if minio_credential:
        from minio import Minio

        with open(minio_credential) as f:
            server, access_key, secret_key = f.readline().split(';')
            secret_key = secret_key.strip()
            minio_client = Minio(server, access_key=access_key, secret_key=secret_key, secure=True)
    else:
        minio_client = None

    torch.manual_seed = np.random.seed = 112358
    data_loader_settings = dict(
        dataset_name=dataset_name,
        select_classes=select_classes,
        dir_path=dir_path,
        num_points=num_points,
        k=k,  # preprocess k index to speed up training (invariant to rotations and scale)
        translation=False,
        rotation=False,
        batch_size=batch_size,
        final=final,
    )
    model_settings = dict(encoder_name=encoder_name,
                          decoder_name=decoder_name,
                          components=components,
                          gf=graph_filtering,
                          cw_dim=cw_dim,
                          z_dim=z_dim,
                          k=k,
                          m=m,
                          ae=ae,
                          book_size=book_size,
                          dim_embedding=dim_embedding
                          )
    model = get_model(**model_settings).to(device).eval()  # set to train by the trainer class later
    if task == 'return model for profiling':
        dummy_input = [torch.ones(batch_size, num_points, 3, device=device),
                       torch.zeros(batch_size, num_points, k, device=device, dtype=torch.long)]
        if ae == 'VQVAE':
            model.recon_cw = True  # profile both encodings
        return model, dummy_input
    elif task == 'return loaded model for random generation':
        assert ae == 'VQVAE', 'Autoencoder does not support realistic cloud generation'
        load_path = os.path.join('models', exp_name, f'model_epoch{training_epochs}.pt')
        assert os.path.exists(load_path), 'No pretrained experiment in ' + load_path
        model.load_state_dict(torch.load(load_path, map_location=device))
        model.decoder.m = m
        return model

    train_loader, val_loader, test_loader = get_loaders(**data_loader_settings)

    lr = {'encoder': initial_learning_rate, 'decoder': initial_learning_rate}
    if ae == 'VQVAE' :
        lr['cw_encoder'] = 10 * initial_learning_rate
    optimizer, optim_args = get_opt(opt_name, lr, weight_decay)

    trainer_settings = dict(
        opt_name=opt_name,
        optimizer=optimizer,
        optim_args=optim_args,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        batch_size=batch_size,
        training_epochs=training_epochs,
        schedule=CosineSchedule(decay_steps=decay_period, min_decay=min_decay),
        minio_client=minio_client,
        recon_loss=recon_loss,
        ae=ae,
        c_reg=c_reg,
    )
    block_args = {**data_loader_settings, **model_settings, **trainer_settings}
    trainer = get_ae_trainer(model, exp_name, block_args)

    if task == 'train cw encoder':
        assert ae == 'VQVAE', 'Only VQVAE supported'
        load_path = os.path.join('models', exp_name, f'model_epoch{training_epochs}.pt')
        assert os.path.exists(load_path), 'No pretrained experiment in ' + load_path
        model_state = torch.load(load_path, map_location=device)
        assert load < 1, 'Only loading the last saved version is supported'

        for name in list(model_state):
            if load == -1:
                # trainer.model.cw_encoder.reset_parameters()
                if name[:10] == 'cw_encoder':
                    model_state.popitem(name)  # temporary feature to experiment with different cw_encoders
        trainer.model.load_state_dict(model_state, strict=False)

        cw_train_loader, cw_test_loader = get_cw_loaders(trainer, m, final)
        optimizer, optim_args = get_opt(opt_name, 10 * initial_learning_rate, weight_decay)
        block_args.update(dict(train_loader=cw_train_loader, val_loader=None, test_loader=cw_test_loader,
                               optimizer=optimizer,
                               optim_args=optim_args,
                               lr=10 * initial_learning_rate))
        cw_trainer = CWTrainer(model, exp_name, block_args)
        if not model_eval:
            while training_epochs > cw_trainer.epoch:
                cw_trainer.train(checkpoint_every)
                cw_trainer.save()
                assert torch.all(torch.isclose(cw_trainer.model.codebook, trainer.model.cw_encoder.codebook))
                cw_trainer.test(partition='test')  # tests on val when not final because val has been saved as test
                trainer.model.load_state_dict(torch.load(load_path, map_location=device))
                with trainer.model.from_z:
                    trainer.test(partition='test' if final else 'val', m=m)
        else:
            cw_trainer.test(partition='test', save_outputs=True)
            trainer.test_cw_recon(partition='test' if final else 'val', m=m, all_metrics=True, denormalise=denormalise)
            from sklearn.decomposition import PCA
            mu = torch.stack(cw_trainer.test_outputs['mu'])
            torch.save(mu, 'mu.pt')
            d_mu = torch.stack(cw_trainer.test_outputs['d_mu'][0])
            torch.save(d_mu, 'd_mu.pt')
            pca = PCA(3)
            cw_pca = pca.fit_transform(mu.numpy())
            from src.plot_PC import pc_show
            pc_show(torch.FloatTensor(cw_pca), colors='blue')
        return
    if task == 'evaluate random generation':
        assert ae == 'VQVAE', 'Only VQVAE supported'
        load_path = os.path.join('models', exp_name, f'model_epoch{training_epochs}.pt')
        assert os.path.exists(load_path), 'No pretrained experiment in ' + load_path
        model_state = torch.load(load_path, map_location=device)
        assert load < 1, 'Only loading the last saved version is supported'
        trainer.model.load_state_dict(model_state)
        with trainer.model.from_z:
            trainer.test(partition='test' if final else 'val', m=m, all_metrics=True, denormalise=denormalise)
        trainer.evaluate_generated_set(m, metric='Emd')
        return
    # loads last model
    if load == 0:
        trainer.load()
    elif load > 0:
        trainer.load(load)

    if task == 'visualise reconstructions':
        trainer.model.decoder.m = m
        dataset = (train_loader if not model_eval else test_loader if final else val_loader).dataset
        input_pcs = []
        recon_pcs = []
        for i in ind:
            assert i < len(dataset), 'Index is too large for the selected dataset'
            dataset_row = dataset[i]
            scale = dataset_row[0][0]
            input_pcs.append(dataset_row[0][1] * scale)
            torch_input = dataset_row[0][-2].unsqueeze(0).to(device)
            torch_input.requires_grad = False
            recon_pcs.append(trainer.model(x=torch_input, indices=None)['recon'][0] * scale)
        return ind, input_pcs, recon_pcs

    # if task == 'evaluate random generation':
    #     trainer.evaluate_generated_set(m, metric='chamfer')
    #     return

    if not model_eval:
        if load == -1 and decoder_name == 'TearingNet':
            exp_name_split = exp_name.split('_')
            exp_name_split[1] = 'FoldingNet'
            load_path = os.path.join('models', '_'.join(exp_name_split), f'model_epoch{training_epochs}.pt')
            assert os.path.exists(load_path), 'No pretrained experiment in ' + load_path
            state_dict = torch.load(load_path, map_location=device)
            trainer.model.load_state_dict(state_dict, strict=False)

        while training_epochs > trainer.epoch:
            if m_training == -1:
                m_training = max(512, (4096 * trainer.epoch) // training_epochs)
                trainer.update_m_training(m_training)
            trainer.train(checkpoint_every)
            trainer.save()
            trainer.test(partition='test' if final else 'val', m=m)
    else:
        trainer.test(partition='test' if final else 'val', m=m, all_metrics=True, denormalise=denormalise)


if __name__ == '__main__':
    main()
