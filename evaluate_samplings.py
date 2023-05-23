import warnings
from src.options import parse_args_and_set_seed
from src.dataset import get_loaders
from src.model import get_model
from src.trainer import get_trainer


def evaluate_samplings():
    args = parse_args_and_set_seed(task='eval_gen', description='Evaluates random samplings')
    assert args.model_head == 'VQVAE' or "Oracle", 'Only VQVAE supported'
    model = get_model(**vars(args))
    train_loader, val_loader, test_loader = get_loaders(**vars(args))
    loaders = dict(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer = get_trainer(model, loaders, args=args)
    test_partition = 'train' if args.eval_train else 'test' if args.final else 'val'
    warnings.simplefilter("error", UserWarning)
    trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    trainer.evaluate_generated_set(test_partition,
                                   repeat_chamfer=args.ch_tests,
                                   repeat_emd=args.emd_tests,
                                   oracle=args.model_head == 'Oracle')


if __name__ == '__main__':
    evaluate_samplings()
