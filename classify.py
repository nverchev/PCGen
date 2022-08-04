import torch
import argparse
import pykeops
from dataset import get_dataset
from optim import get_opt, CosineSchedule
from trainer import get_vae_trainer

pykeops.set_verbose(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Point Cloud Encoder - Generator')

    parser.add_argument('--model', type=str, default='DGCNN', choices=["DGCNN"])
    parser.add_argument('--loss', type=str, default='CE', choices=["CE"],
                        help='reconstruction loss')
    parser.add_argument('--experiment', type=str, default='',
                        help='Name of the experiment. If it starts with "final" the test set is used for eval.')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40', 'shapenet'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--load', type=int, default=0,
                        help='load a saved model with the same settings. -1 for starting from scratch,'
                             '0 for most recent, otherwise epoch after which the model was saved')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=["SGD", "SGD_nesterov", "Adam", "AdamW"],
                        help='SGD has no momentum, otherwise momentum = 0.9')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--cuda', type=bool, default=True, help='enables CUDA training')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model (exp_name needs to start with "final")')
    parser.add_argument('--k', type=int, default=20,
                        help='number of neighbours of a point (counting the point itself) in DGCNN]')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points of the training dataset [currently fixed]')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Default is given by model_exp_name')
    parser.add_argument('--dir_path', type=str, default='./', help='Directory for storing data and models')
    parser.add_argument('--minio_credential', type=str, default='',
                        help='path of file with written server;access_key;secret_key')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    encoder_name = args.encoder
    decoder_name = args.decoder
    model_name = encoder_name + "_" + decoder_name
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    dataset = args.dataset
    recon_loss = args.recon_loss
    experiment = args.experiment
    dir_path = args.dir_path
    training_epochs = args.epochs
    opt_name = args.optimizer
    batch_size = args.batch_size
    load = args.load
    initial_learning_rate = args.lr
    weight_decay = args.wd
    model_eval = args.eval
    k = args.k
    num_points = args.num_points
    minio_credential = args.minio_credential
    m_training = args.m_training
    final = experiment[:5] == 'final'
    if minio_credential:
        from minio import Minio

        with open(minio_credential) as f:
            server, access_key, secret_key = f.readline().split(';')
            secret_key = secret_key.strip()
            minioClient = Minio(server, access_key=access_key, secret_key=secret_key, secure=True)
    else:
        minioClient = None
    exp_name = '_'.join([model_name, recon_loss, experiment]) if args.model_path == "" else args.model_path

    train_loader, val_loader, test_loader = get_dataset(experiment, dataset, batch_size, dir_path=dir_path,
                                                        n_points=num_points)
    model = VAE(encoder_name, decoder_name, k=k)
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
        'schedule': CosineSchedule(decay_steps=training_epochs, min_decay=0.1),
        'minioClient': minioClient,
        'dir_path': dir_path,
    }

    trainer = get_vae_trainer(model, recon_loss, exp_name, block_args)
    for k, v in block_args.items():
        if not isinstance(v, (type, torch.utils.data.dataloader.DataLoader)):
            print(k, ': ', v)

    # loads last model
    if load == 0:
        trainer.load()
    elif load > 0:
        trainer.load(load)

    if not model_eval:
        while training_epochs >= trainer.epoch:
            if m_training == 0:
                m = max(128, (4096 * trainer.epoch) // training_epochs)
            else:
                m = m_training
            trainer.update_m_training(m)
            trainer.train(10)
            print(trainer.model.encode.conv)
            if not final:
                trainer.clas_metric()
            trainer.save()

    trainer.clas_metric(final=final)
