from src.options import parse_args_and_set_seed
from src.dataset import get_loaders, EmptyDataset
from src.model import get_model
from src.trainer import get_trainer


def eval_model():
    args = parse_args_and_set_seed(description='Eval a loaded model')
    model = get_model(**vars(args))
    train_loader, val_loader, test_loader = get_loaders(**vars(args))
    loaders = dict(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer = get_trainer(model, loaders, args=args)
    test_partition = 'test' if args.final else 'val'
    trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    if not args.final:
        trainer.plot_loss_metric(partition='train and val', start_from=trainer.epoch - 50)
    trainer.test(partition=test_partition, all_metrics=True, de_normalise=args.de_normalise)


if __name__ == '__main__':
    eval_model()
