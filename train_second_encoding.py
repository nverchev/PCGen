from src.options import parse_process_args_and_set_seed
from src.trainer import get_trainer, get_cw_trainer


def train_second_encoding():
    args = parse_process_args_and_set_seed(task='train_vae', description='Train second encoding')
    assert args.model_head == 'VQVAE', 'Only VQVAE supported'
    vqvae_trainer = get_trainer(args)
    vqvae_trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    vqvae_trainer.epoch = args.epochs
    cw_trainer = get_cw_trainer(vqvae_trainer, args)
    if not args.vae_load:
        while args.vae_epochs > cw_trainer.epoch:
            cw_trainer.train(args.vae_checkpoint, val_after_train=True)
            if args.training_plot:
                # setting start > 0 is a quicker and automatic alternative than zooming the plotly plot
                start = max(cw_trainer.epoch - 10 * args.checkpoint, 0)
                cw_trainer.plot_learning_curves(start=start, title='w-encoding')
                cw_trainer.plot_learning_curves(start=start, title='KLD', loss_or_metric='KLD')
                cw_trainer.plot_learning_curves(start=start, title='Accuracy', loss_or_metric='Accuracy')
        cw_trainer.save()
        if args.training_plot:
            cw_trainer.plot_learning_curves(title='w-encoding')
            cw_trainer.plot_learning_curves(title='KLD', loss_or_metric='KLD')
            cw_trainer.plot_learning_curves(title='Accuracy', loss_or_metric='Accuracy')
    cw_trainer.test(args.test_partition, all_metrics=True, de_normalize=args.de_normalize, save_outputs=True)
    cw_trainer.show_latent()


if __name__ == '__main__':
    train_second_encoding()
