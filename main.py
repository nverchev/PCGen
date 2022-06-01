import torch
import argparse
import pykeops
from dataset import get_dataset
from optim import get_opt, CosineSchedule
from trainer import get_trainer
from VAE import get_vae

pykeops.set_verbose(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Encoder - Generator')

    parser.add_argument('--model', type=str, default='VAE_Gen', choices=["BaseVAE", "PointNet", "VAE_Gen", "PCTVAE"],
                        help='architecture')
    parser.add_argument('--recon_loss', type=str, default='Chamfer', choices=["Chamfer", "Sinkhorn", "NLL", "MMD"],
                        help='reconstruction loss')
    parser.add_argument('--exp_name', type=str, default='',
                        help='Name of the experiment. If it starts with "final" the test set is used for eval.')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'],
                        help="Currently only one dataset available")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=["SGD", "SGD_nesterov", "Adam", "AdamW"]
                        , help='SGD has no momentum, otherwise momentum = 0.9')
    parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
    parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
    parser.add_argument('--cuda', type=bool, default=True,  help='enables CUDA training')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model (exp_name needs to start with "final")')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points of the training dataset [currently fixed]')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Default is given by model_recon_loss_exp_name')
    args = parser.parse_args()

    model_name = args.model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") and args.cuda
    recon_loss = args.recon_loss
    experiment = args.experiment
    final = experiment[:5] == 'final'
    training_epochs = args.epochs
    opt_name = args.optimizer
    batch_size = args.batch_size
    initial_learning_rate = args.lr
    weight_decay = args.wd
    model_eval = args.eval
    num_points = args.num_points
    exp_name = '_'.join([model_name, recon_loss, experiment]) if args.model_path is "" else args.model_path

    train_loader, val_loader, train_val_loader, test_loader = get_dataset(batch_size)
    model = get_vae(model_name)
    optimizer, optim_args = get_opt(opt_name, initial_learning_rate, weight_decay)
    train_loader = train_loader if not final else train_val_loader
    val_loader = val_loader if not final else None
    block_args = {
        'optim_name': opt_name,
        'optim': optimizer[opt_name],
        'optim_args': optim_args[opt_name],
        'train_loader': train_loader,
        'device': device,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'batch_size': batch_size,
        'schedule': CosineSchedule(decay_steps=training_epochs, min_decay=0.1)
    }

    trainer = get_trainer(model, exp_name, recon_loss, block_args)
    for k, v in block_args.items():
        if not isinstance(v, (type, torch.utils.data.dataloader.DataLoader)):
            print(k, ': ', v)

    if not model_eval:
        for _ in range(training_epochs // 10):
            trainer.train(10)
            if not final:
                trainer.test(on="val", m=512)
            trainer.save()

    if final:
        trainer.test(on=final)
