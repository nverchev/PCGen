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
    parser.add_argument('--decoder', type=str, default='PCGen', choices=['PCGen', 'Full', 'FoldingNet',
                                                                         'TearingNet', 'AtlasNet'])
    parser.add_argument('--exp', type=str, default='',
                        help='Name of the experiment. If it starts with "final" the test set is used for eval.')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Default is given by "_".join(model(GF), recon_loss, exp)')
    parser.add_argument('--gf', action='store_true', default=False, help='Graph filtering after decoder')
    parser.add_argument('--recon_loss', type=str, default='Chamfer',
                        choices=['Chamfer', 'ChamferA', 'ChamferS', 'Sinkhorn'], help='PC reconstruction loss')
    parser.add_argument('--ae', type=str, default='AE', choices=['AE', 'VQVAE'], help='VQVAE adds quantisation')
    parser.add_argument('--dir_path', type=str, default='./', help='Directory for storing data and models')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40', 'shapenet', 'coins'])
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Number of points of the training dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optim', type=str, default='AdamW', choices=['SGD', 'SGD_momentum', 'Adam', 'AdamW'],
                        help='SGD_momentum has momentum = 0.9')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.000001, help='Wweight decay')
    parser.add_argument('--k', type=int, default=20,
                        help='Number of neighbours of a point (counting the point itself) in DGCNN]')
    parser.add_argument('--cw_dim', type=int, default=512, help='Dimension of the codeword space')
    parser.add_argument('--book_size', type=int, default=16, help='Dictionary size for vector quantisation')
    parser.add_argument('--dim_embedding', type=int, default=4, help='Dimension of the vector for vector quantisation')
    parser.add_argument('--c_reg', type=float, default=1, help='Coefficient for regularization')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Runs on CPU')
    parser.add_argument('--epochs', type=int, default=250, help='Number of total training epochs')
    parser.add_argument('--checkpoint', type=int, default=10, help='Number of epochs between checkpoints')
    parser.add_argument('--m_training', type=int, default=2048,
                        help='Points generated when training, 0 for  increasing sequence 128 -> 4096 ')
    parser.add_argument('--m_test', type=int, default=2048, help='Points generated when testing')
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
                    'train cw encoder']
    args = parse_args()
    encoder_name = args.encoder
    decoder_name = args.decoder
    graph_filtering = args.gf
    experiment = args.exp
    recon_loss = args.recon_loss
    ae = args.ae
    gf = 'GF' if graph_filtering else ''
    exp_name = args.model_path or '_'.join([encoder_name, decoder_name + gf, recon_loss, ae, experiment])
    final = experiment[:5] == 'final'
    dir_path = args.dir_path
    dataset_name = args.dataset
    num_points = args.num_points
    batch_size = args.batch_size
    k = args.k
    opt_name = args.optim
    initial_learning_rate = args.lr
    weight_decay = args.wd
    cw_dim = args.cw_dim
    book_size = args.book_size
    dim_embedding = args.dim_embedding
    c_reg = args.c_reg
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    training_epochs = args.epochs
    checkpoint_every = args.checkpoint
    m = args.m_test
    m_training = args.m_training
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

    torch.manual_seed = 112358
    np.random.seed = 112358
    data_loader_settings = dict(
        dataset_name=dataset_name,
        dir_path=dir_path,
        num_points=num_points,
        # preprocess k index to speed up training (invariant to rotations and scale)
        k=k,
        translation=False,
        rotation=True,
        batch_size=batch_size,
        final=final,
    )
    model_settings = dict(encoder_name=encoder_name,
                          decoder_name=decoder_name,
                          gf=graph_filtering,
                          cw_dim=cw_dim,
                          k=k,
                          m=m,
                          ae=ae,
                          book_size=book_size,
                          dim_embedding=dim_embedding

                          )
    model = get_model(**model_settings).to(device).eval()  # set to train by the trainer class later
    if task == 'return model for profiling':
        dummy_input = [torch.ones(batch_size, num_points, 3, device=device),
                       torch.ones(batch_size, num_points, k, device=device, dtype=torch.long)]
        return model, dummy_input
    elif task == 'return loaded model for random generation':
        assert ae == "VQVAE", "Autoencoder does not support realistic cloud generation"
        load_path = os.path.join(dir_path, 'models', exp_name, f'model_epoch{training_epochs}.pt')
        assert os.path.exists(load_path), "No pretrained experiment in " + load_path
        model.load_state_dict(torch.load(load_path, map_location=device))
        z_dim = cw_dim // 64
        z = torch.randn(batch_size, z_dim).to(device)
        t = model.cw_encoder.codebook[torch.randint(model.cw_encoder.book_size, [batch_size])]
        return model, z, t

    train_loader, val_loader, test_loader = get_loaders(**data_loader_settings)
    optimizer, optim_args = get_opt(opt_name, initial_learning_rate, weight_decay)
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
        schedule=CosineSchedule(decay_steps=training_epochs, min_decay=0.1),
        minio_client=minio_client,
        model_path=os.path.join(dir_path, 'model'),
        recon_loss=recon_loss,
        ae=ae,
        c_reg=c_reg,
    )

    block_args = {**data_loader_settings, **model_settings, **trainer_settings}
    trainer = get_ae_trainer(model, exp_name, block_args)
    if task == 'train cw encoder':
        assert ae == "VQVAE", "Only VQVAE supported"
        #trainer.load()  # cw encoder is embedded to the last epoch of the VQVAE model
        load_path = os.path.join(dir_path, 'models', exp_name, f'model_epoch{training_epochs}.pt')
        assert os.path.exists(load_path), "No pretrained experiment in " + load_path
        model_state = torch.load(load_path, map_location=device)
        model_state.popitem("*cw_encoder*")  # temporary feature to experiment with different cw_encoders
        model.load_state_dict(model_state, strict=False)
        assert load < 1, "Only loading the last saved version is supported"
        if load == -1:
            trainer.model.cw_encoder.reset_parameters()

        cw_train_loader, cw_test_loader = get_cw_loaders(trainer, final)
        block_args.update(dict(train_loader=cw_train_loader, val_loader=None, test_loader=cw_test_loader))
        cw_trainer = CWTrainer(model, exp_name, block_args)
        if not model_eval:
            while training_epochs > cw_trainer.epoch:
                cw_trainer.train(checkpoint_every)
                cw_trainer.save()
                cw_trainer.test(partition='test')  # tests on val when not final because val has been saved as test
                trainer.test_cw_recon(partition='test' if final else 'val')
        else:
            cw_trainer.test(partition='test')
            trainer.test_cw_recon(partition='test' if final else 'val')
        return

    # loads last model
    if load == 0:
        trainer.load()
    elif load > 0:
        trainer.load(load)

    if not model_eval:
        if load == -1 and decoder_name == 'TearingNet':
            exp_name_split = exp_name.split('_')
            exp_name_split[1] = 'FoldingNet'
            load_path = os.path.join(dir_path, 'models', '_'.join(exp_name_split), f'model_epoch{training_epochs}.pt')
            assert os.path.exists(load_path), "No pretrained experiment in " + load_path
            state_dict = torch.load(load_path, map_location=device)
            trainer.model.load_state_dict(state_dict, strict=False)

        while training_epochs > trainer.epoch:
            if m_training == 0:
                m_training = max(512, (4096 * trainer.epoch) // training_epochs)
                trainer.update_m_training(m_training)
            trainer.train(checkpoint_every)
            trainer.save()
            trainer.test(partition='test' if final else 'val')
    else:
        trainer.test(partition='test' if final else 'val')


if __name__ == '__main__':
    main()
