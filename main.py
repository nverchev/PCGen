import torch
import argparse
import pykeops
from dataset import get_dataset
from optim import get_opt, CosineSchedule
from trainer import get_trainer
from VAE import VAE

pykeops.set_verbose(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud Encoder - Generator')

    parser.add_argument('--encoder', type=str, default='DGCNN', choices=["MLP", "PointNet", "DGCNN", "PCT"])
    parser.add_argument('--decoder', type=str, default='Gen', choices=["MLP", "Gen, FoldingNet"])
    parser.add_argument('--recon_loss', type=str, default='Chamfer', choices=["Chamfer", "Sinkhorn", "NLL", "MMD"],
                        help='reconstruction loss')
    parser.add_argument('--experiment', type=str, default='',
                        help='Name of the experiment. If it starts with "final" the test set is used for eval.')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'],
                        help="Currently only one dataset available")
    parser.add_argument('--download', type=str, default='do_not_download',
                        choices=["from_zip", "from_minio", "do_not_download"],
                        help="You can process the dataset from the zip file. Otherwise you can  \
                                        download an already processed one from a server using minio")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=["SGD", "SGD_nesterov", "Adam", "AdamW"]
                        , help='SGD has no momentum, otherwise momentum = 0.9')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.000, help='weight decay')
    parser.add_argument('--cuda', type=bool, default=True, help='enables CUDA training')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model (exp_name needs to start with "final")')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points of the training dataset [currently fixed]')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Default is given by model_recon_loss_exp_name')
    parser.add_argument('--dir_path', type=str, default='./', help='Directory for storing data and models')
    parser.add_argument('--minio_credential', type=str, default='',
                        help='path of file with written server.access_key.secret_key')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    encoder_name = args.encoder
    decoder_name = args.decoder
    model_name = encoder_name + "_" + decoder_name
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    recon_loss = args.recon_loss
    experiment = args.experiment
    dir_path = args.dir_path
    training_epochs = args.epochs
    opt_name = args.optimizer
    batch_size = args.batch_size
    initial_learning_rate = args.lr
    weight_decay = args.wd
    model_eval = args.eval
    num_points = args.num_points
    download = args.download
    minio_credential = args.minio_credential
    if minio_credential:
        from minio import Minio

        with open(minio_credential) as f:
            server, access_key, secret_key = f.readline().split(';')
            secret_key = secret_key.strip()
            minioClient = Minio(server, access_key=access_key, secret_key=secret_key, secure=True)
    else:
        minioClient = None
    exp_name = '_'.join([model_name, recon_loss, experiment]) if args.model_path == "" else args.model_path

    train_loader, val_loader, test_loader = get_dataset(experiment, batch_size, dir_path=dir_path, download=download,
                                                        minioClient=minioClient, n_points=num_points)
    model = VAE(encoder_name, decoder_name)
    optimizer, optim_args = get_opt(opt_name, initial_learning_rate, weight_decay)
    block_args = {
        'optim_name': opt_name,
        'optim': optimizer,
        'optim_args': optim_args,
        'train_loader': train_loader,
        'device': device,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'batch_size': batch_size,
        'schedule': CosineSchedule(decay_steps=training_epochs, min_decay=0.01),
        'minioClient': minioClient,
        'dir_path': dir_path,
    }

    trainer = get_trainer(model, recon_loss, exp_name, block_args)
    for k, v in block_args.items():
        if not isinstance(v, (type, torch.utils.data.dataloader.DataLoader)):
            print(k, ': ', v)

    if not model_eval:
        m = 128
        for _ in range(training_epochs // 10):
            trainer.update_m_training(m)
            m *= 2
            trainer.train(10)
            if experiment[:5] != 'final':
                trainer.clas_metric()
            trainer.save()

    # loads last model
    trainer.load()
    if experiment[:5] == 'final':
        trainer.clas_metric(final=True)
    else:
        trainer.clas_metric()

