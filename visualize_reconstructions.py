import warnings
import torch
from src.options import parse_args_and_set_seed
from src.dataset import get_loaders
from src.model import get_model
from src.trainer import get_trainer
from src.viz_pc import  render_cloud


def visualize_reconstruction():
    args = parse_args_and_set_seed(task='viz', description='visualise and render reconstructions')
    model = get_model(**vars(args))
    train_loader, val_loader, test_loader = get_loaders(**vars(args))
    loaders = dict(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer = get_trainer(model, loaders, args=args)
    if args.model_head != 'Oracle':
        warnings.simplefilter("error", UserWarning)
        trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    dataset = (train_loader if args.eval_train else test_loader if args.final else val_loader).dataset
    if args.model_head == 'VQVAE' and args.viz_double_encoding:
        trainer.model.double_encoding = True
    for i in args.viz:
        assert i < len(dataset), 'Index is too large for the selected dataset'
        dataset_row = dataset[i]
        scale = dataset_row[0][0]
        pc_in = dataset_row[0][1] * scale
        torch_input = dataset_row[0][-2].unsqueeze(0).to(args.device)
        with torch.inference_mode():
            trainer.model.eval()
            pc_out = trainer.model(x=torch_input, indices=None)['recon'][0] * scale
        render_cloud([pc_in.detach().cpu()], name=f'sample_{i}.png')
        render_cloud([pc_out.detach().cpu()], name=f'reconstruction_{i}.png')


if __name__ == '__main__':
    visualize_reconstruction()
