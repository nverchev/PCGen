import warnings
from src.options import parse_args_and_set_seed
from src.dataset import get_loaders
from src.model import get_model
from src.trainer import get_trainer


def eval_model():
    args = parse_args_and_set_seed(description='Eval a loaded model')
    model = get_model(**vars(args))
    train_loader, val_loader, test_loader = get_loaders(**vars(args))
    loaders = dict(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer = get_trainer(model, loaders, args=args)
    test_partition = 'train' if args.eval_train else 'test' if args.final else 'val'
    if args.model_head != 'Oracle':
        warnings.simplefilter("error", UserWarning)
        trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    trainer.test(partition=test_partition, all_metrics=True, de_normalize=args.de_normalize)
    if args.training_plot:
        trainer.plot_loss_metric(start=args.checkpoint, loss_metric='Chamfer')


if __name__ == '__main__':
    eval_model()
