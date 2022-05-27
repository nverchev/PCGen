import torch
import pykeops
from dataset import get_dataset
from optim import get_opt, CosineSchedule
from trainer import get_trainer
from VAE import get_vae

pykeops.set_verbose(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model_name = "Base_VAE"
    recon_loss = "Chamfer"
    version = ""
    opt_name = "AdamW"
    batch_size = 64
    initial_learning_rate = 0.001
    weight_decay = 0.00001
    train_loader, val_loader, train_val_loader, test_loader = get_dataset(batch_size)
    model = get_vae(model_name)
    optimizer, optimi_args = get_opt(opt_name, initial_learning_rate, weight_decay)
    train_loader = train_loader if version[:5] != 'final' else train_val_loader
    val_loader = val_loader if version[:5] != 'final' else None
    block_args = {
        'optim_name': opt_name,
        'optim': optimizer[opt_name],
        'optim_args': optimi_args[opt_name],
        'train_loader': train_loader,
        'device': device,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'batch_size': batch_size,
        'schedule': CosineSchedule()
    }
    trainer_name = '_'.join([model_name, recon_loss, version])
    trainer = get_trainer(model, trainer_name, recon_loss, block_args)
    for k, v in block_args.items():
        if not isinstance(v, (type, torch.utils.data.dataloader.DataLoader)):
            print(k, ': ', v)
    for _ in range(6):
        trainer.train(10)
        trainer.test(on="val", m=512)
        trainer.save()
