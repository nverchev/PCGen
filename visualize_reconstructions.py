import warnings
import torch
from src.options import parse_args_and_set_seed
from src.dataset import get_loaders
from src.model import get_model
from src.trainer import get_trainer
from src.viz_pc import render_cloud, infer_and_visualize


def visualize_reconstruction():
    args = parse_args_and_set_seed(task='viz', description='visualise and render reconstructions')
    # model = get_model(**vars(args))
    train_loader, val_loader, test_loader = get_loaders(**vars(args))
    # loaders = dict(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    # trainer = get_trainer(model, loaders, args=args)
    # if args.model_head != 'Oracle':
    #     warnings.simplefilter("error", UserWarning)
    #     trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    dataset = (train_loader if args.eval_train else test_loader if args.final else val_loader).dataset
    # if args.model_head == 'VQVAE' and args.viz_double_encoding:
    #     trainer.model.double_encoding = True
    input_pcs = []
    for i in range(len(dataset)): #args.viz:
        assert i < len(dataset), 'Index is too large for the selected dataset'
        dataset_row = dataset
        input_pc = dataset_row[i][0][1]
        render_cloud([input_pc], name=f'sample_{i}.png', interactive=args.interactive_plot)
    #     input_pcs.append(input_pc)
    # input_pcs = torch.stack(input_pcs).to(args.device)
    # infer_and_visualize(model, args, input_pc=input_pcs, n_clouds=len(args.viz))


if __name__ == '__main__':
    visualize_reconstruction()
