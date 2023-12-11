import warnings
from src.options import parse_process_args_and_set_seed
from src.trainer import get_trainer


def evaluate_samplings():
    args = parse_process_args_and_set_seed(task='eval_gen', description='Evaluates random samplings')
    assert args.model_head == 'VQVAE' or "Oracle", 'Only VQVAE supported'
    trainer = get_trainer(args=args)
    warnings.simplefilter("error", UserWarning)
    trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    trainer.evaluate_generated_set(args.test_partition,
                                   repeat_chamfer=args.ch_tests,
                                   repeat_emd=args.emd_tests,
                                   oracle=args.model_head == 'Oracle')


if __name__ == '__main__':
    evaluate_samplings()
