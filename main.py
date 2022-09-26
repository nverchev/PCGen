import os
import torch
import numpy as np
import argparse
import pykeops
from src.dataset import get_dataset
from src.optim import get_opt, CosineSchedule
from src.trainer import get_vae_trainer
from src.model import get_model

pykeops.set_verbose(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud Encoder - Generator')

    parser.add_argument('--encoder', type=str, default='LDGCNN', choices=['LDGCNN', 'DGCNN', 'FoldingNet'])
    parser.add_argument('--decoder', type=str, default='PCGen', choices=['PCGen', 'Full', 'FoldingNet',
                                                                         'TearingNet', 'AtlasNet'])
    parser.add_argument('--experiment', type=str, default='',
                        help='Name of the experiment. If it starts with "final" the test set is used for eval.')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Default is given by model_recon_loss_exp_name')
    parser.add_argument('--gf', action='store_true', default=False, help='graph filtering after decoder')
    parser.add_argument('--recon_loss', type=str, default='Chamfer',
                        choices=['Chamfer', 'ChamferA', 'ChamferS', 'Sinkhorn'], help='reconstruction loss')
    parser.add_argument('--vae', type=str, default='NoVAE',
                        choices=['NoVAE', 'VAE', 'VQVAE'], help='type of regularization')
    parser.add_argument('--dir_path', type=str, default='./', help='Directory for storing data and models')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40', 'shapenet', 'coins'])
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points of the training dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optim', type=str, default='AdamW', choices=['SGD', 'SGD_momentum', 'Adam', 'AdamW'],
                        help='SGD_momentum, has momentum = 0.9')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.000001, help='weight decay')
    parser.add_argument('--k', type=int, default=20,
                        help='number of neighbours of a point (counting the point itself) in DGCNN]')
    parser.add_argument('--cw_dim', type=int, default=512, help='dimension of the latent space')
    parser.add_argument('--dict_size', type=int, default=16, help='dictionary size for vector quantisation')
    parser.add_argument('--dim_embedding', type=int, default=4, help='dim of the vector for vector quantisation')
    parser.add_argument('--c_reg', type=float, default=1, help='coefficient for regularization')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='runs on CPU')
    parser.add_argument('--epochs', type=int, default=250, help='number of total training epochs')
    parser.add_argument('--checkpoints', type=int, default=10, help='number of epochs between checkpoints')
    parser.add_argument('--m_training', type=int, default=2048,
                        help='Points generated when training, 0 for  increasing sequence 128 -> 4096 ')
    parser.add_argument('--m_test', type=int, default=2048, help='Points generated when testing')
    parser.add_argument('--load', type=int, default=-1,
                        help='load a saved model with the same settings. -1 for starting from scratch,'
                             '0 for most recent, otherwise epoch after which the model was saved')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model (exp_name needs to start with "final")')
    parser.add_argument('--minio_credential', type=str, default='',
                        help='path of file with written server;access_key;secret_key')
    return parser.parse_args()


def main(task='train/eval'):
    args = parse_args()
    encoder_name = args.encoder
    decoder_name = args.decoder
    graph_filtering = args.gf
    experiment = args.experiment
    recon_loss = args.recon_loss
    vae = args.vae
    gf = 'GF' if graph_filtering else ''
    exp_name = args.model_path or '_'.join([encoder_name, decoder_name + gf, recon_loss, vae, experiment])
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
    dict_size = args.dict_size
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
        # preprocess k index to speed up training (invariant to affine transformations)
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
                          vae=vae,
                          dict_size=dict_size,
                          dim_embedding=dim_embedding

                          )
    model = get_model(**model_settings)
    if task == 'return model for profiling':
        dummy_input = [torch.ones(batch_size, num_points, 3, device=device),
                       torch.ones(batch_size, num_points, k, device=device, dtype=torch.long)]
        return model.to(device), dummy_input
    elif task == 'return loaded model':
        assert vae == "VQVAE", "Autoencoder does not support realistic cloud generation"
        load_path = os.path.join(dir_path, 'models', exp_name, f'model_epoch{training_epochs}.pt')
        assert os.path.exists(load_path), "No pretrained experiment in " + load_path
        model.load_state_dict(torch.load(load_path, map_location=device))
        z_dim = cw_dim // 64
        z = torch.randn(batch_size, z_dim).to(device)
        return model.to(device), z

    train_loader, val_loader, test_loader = get_dataset(**data_loader_settings)
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
        schedule=CosineSchedule(decay_steps=training_epochs, min_decay=0.1),
        minio_client=minio_client,
        model_path=os.path.join(dir_path, 'model'),
        recon_loss=recon_loss,
        vae=vae,
        c_reg=c_reg,
    )

    block_args = {**data_loader_settings, **model_settings, **trainer_settings}
    trainer = get_vae_trainer(model, exp_name, block_args)
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
