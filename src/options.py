import argparse
import os
import yaml
import torch
import numpy as np
from argparse import ArgumentTypeError, BooleanOptionalAction


# type with value restrictions
def bounded_num(numeric_type, v_min=None, v_max=None):
    def check_lower_bound(value):
        try:
            number = numeric_type(value)
            if v_min is not None and number < v_min:
                raise ArgumentTypeError(f'Input {value} must be greater than {v_min}.')
            if v_max is not None and v_max < number:
                raise ArgumentTypeError(f'Input {value} must be smaller than {v_min}.')
        except ValueError:
            raise ArgumentTypeError(f'Input incompatible with type {numeric_type.__name__}')
        except TypeError:
            raise ArgumentTypeError(f'{numeric_type.__name__} does not support inequalities')
        return number

    return check_lower_bound


# Parser options
def parser_add_arguments(parser):
    parser.add_argument('--exp', type=str, help='name of the model directory: setup_exp(_final)', default='')

    # dataset options
    dataset_opt = parser.add_argument_group('Dataset options')
    dataset_opt.add_argument('--data_dir', type=str, help='directory for data')
    dataset_opt.add_argument('--final', action=BooleanOptionalAction,
                             help='uses val dataset for training and test dataset for testing, otherwise test on val')
    dataset_opt.add_argument('--dataset_name',
                             choices=['Modelnet40', 'ShapenetAtlas', 'Coins', 'Faust', 'ShapenetFlow'])
    dataset_opt.add_argument('--select_classes', nargs='+', choices=['airplane', 'car', 'chair'],
                             help='select specific classes of the dataset ShapenetFlow')
    dataset_opt.add_argument('--input_points', type=bounded_num(float, v_min=0),
                             help='(maximum) points of the training dataset')
    dataset_opt.add_argument('--translation', action=BooleanOptionalAction, help='random translating training inputs')
    dataset_opt.add_argument('--rotation', action=BooleanOptionalAction, help='random rotating training inputs')
    dataset_opt.add_argument('--resample', type=str, help='two different samplings for input and reference')

    # model options
    model_opt = parser.add_argument_group('Model options')
    dataset_opt.add_argument('--model_pardir', type=str, help='directory for models')
    dataset_opt.add_argument('--model_head', choices=['VQVAE', 'AE', 'Oracle'],
                             help='regularized /  non-regularized / identity. Oracle gives a different resampling of'
                                  'the input for reconstruction and training examples for random generation')
    model_opt.add_argument('--encoder_name', choices=['LDGCNN', 'DGCNN', 'FoldingNet'], help='PC encoder')
    model_opt.add_argument('--decoder_name', choices=['PCGen', 'PCGenC', 'Full', 'FoldingNet', 'TearingNet',
                                                      'AtlasNetDeformation', 'AtlasNetStructures'], help='PC decoder')
    model_opt.add_argument('--components', type=bounded_num(int, v_min=1),
                           help='components of PCGenC or patches in AtlasNet')
    model_opt.add_argument('--filtering', action=BooleanOptionalAction, help='Laplacian filtering after decoder')
    model_opt.add_argument('--k', type=bounded_num(int, v_min=1),
                           help='number of neighbours of a point (counting the point itself) in (L)DGCNN]')
    model_opt.add_argument('--cw_dim', type=bounded_num(int, v_min=1), help='codeword length')

    # training options
    train_opt = parser.add_argument_group('Training options')
    train_opt.add_argument('--recon_loss', choices=['Chamfer', 'ChamferEMD'], help='ChamferEMD adds both')
    train_opt.add_argument('--load', action=BooleanOptionalAction,
                           help='load a saved model from the model directory. See --exp and --load_checkpoint')
    train_opt.add_argument('--load_checkpoint', type=bounded_num(int, v_min=1),
                           help='specify which checkpoint (i.e. training epochs) to load. Default: last one')
    train_opt.add_argument('--batch_size', type=bounded_num(int, v_min=1))
    train_opt.add_argument('--m_training', type=bounded_num(int, v_min=4),
                           help='points generated in training, 0 for input number of points')
    train_opt.add_argument('--m_test', type=bounded_num(int, v_min=4),
                           help='points generated in testing, 0 for input number of points')
    train_opt.add_argument('--opt_name', default='AdamW', choices=['SGD', 'SGD_momentum', 'Adam', 'AdamW'],
                           help='optimizer. SGD_momentum has momentum = 0.9')
    train_opt.add_argument('--lr', type=bounded_num(float, v_min=0), help='learning rate')
    train_opt.add_argument('--wd', type=bounded_num(float, v_min=0), help='weight decay')
    train_opt.add_argument('--min_decay', type=float, help='fraction of the initial lr at the end of train')
    train_opt.add_argument('--epochs', type=bounded_num(int, v_min=1), help='number of total training epochs')
    train_opt.add_argument('--decay_steps', type=bounded_num(int, v_min=0), help='epochs before lr decays stops')
    train_opt.add_argument('--checkpoint', type=bounded_num(int, v_min=1),
                           help='epochs between checkpoints (should divide epochs)')

    # evaluation options
    eval_opt = parser.add_argument_group('Evaluation options')
    eval_opt.add_argument('--eval_train', action=BooleanOptionalAction, help='eval on train instead of test or val')
    eval_opt.add_argument('--viz', type=bounded_num(int, v_min=0), nargs='+', help='sample indices to visualise')
    eval_opt.add_argument('--training_plot', action=BooleanOptionalAction, help='visualize learning curves')
    eval_opt.add_argument('--interactive_plot', action=BooleanOptionalAction, help='3D plot with plotly')
    eval_opt.add_argument('--de_normalize', action=BooleanOptionalAction,
                          help='compare reconstruction and input on coordinates before normalization')

    # utility options
    util_opt = parser.add_argument_group('Utility options')
    util_opt.add_argument('--cuda', action=BooleanOptionalAction, help='run on Cuda')
    util_opt.add_argument('--seed', type=bounded_num(int, v_min=1), help='torch/numpy seed (0 no seed)')


def parser_add_vqvae_arguments(parser):
    vqvae_opt = parser.add_argument_group('VQVAE options', 'ignored when VQVAE is not used')
    vqvae_opt.add_argument('--book_size', type=bounded_num(int, v_min=1), help='dictionary size')
    vqvae_opt.add_argument('--embedding_dim', type=bounded_num(int, v_min=1), help='Codes length')
    vqvae_opt.add_argument('--z_dim', type=bounded_num(int, v_min=1), help='continuous latent space dim')
    vqvae_opt.add_argument('--c_commitment', type=bounded_num(float, v_min=0), help='coefficient for commitment loss')
    vqvae_opt.add_argument('--c_embedding', type=bounded_num(float, v_min=0), help='coefficient for embedding loss')
    vqvae_opt.add_argument('--vq_ema_update', action=BooleanOptionalAction, help='EMA update on quantized codes')
    vqvae_opt.add_argument('--gen', type=bounded_num(int, v_min=1), help='number of generated samples')

    # VAE only
    vae_opt = parser.add_argument_group('Second encoding options', 'Only used when training the discrete variable vae')
    vae_opt.add_argument('--c_kld', type=bounded_num(float, v_min=0), help='Kullback-Leibler Divergence coefficient')
    vae_opt.add_argument('--vae_load', action=BooleanOptionalAction, help='load weights stored in the larger model')
    vae_opt.add_argument('--vae_batch_size', type=bounded_num(int, v_min=1))
    vae_opt.add_argument('--vae_opt_name', default='AdamW', choices=['SGD', 'SGD_momentum', 'Adam', 'AdamW'],
                         help='SGD_momentum has momentum = 0.9')
    vae_opt.add_argument('--vae_lr', type=bounded_num(float, v_min=0), help='learning rate')
    vae_opt.add_argument('--vae_wd', type=bounded_num(float, v_min=0), help='weight decay')
    vae_opt.add_argument('--vae_min_decay', type=float, help='fraction of the initial lr at the end of train')
    vae_opt.add_argument('--vae_epochs', type=bounded_num(int, v_min=1), help='number of total training epochs')
    vae_opt.add_argument('--vae_decay_period', type=bounded_num(int, v_min=0), help='epochs before lr decays stops')
    vae_opt.add_argument('--vae_checkpoint', type=bounded_num(int, v_min=1), help='epochs between checkpoints')


def parse_args_and_set_seed(description='Shared options for training, evaluating and random generation'):
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(title='Experiment default settings',
                                       description='Name of a file in the setups folder without extension',
                                       help='create a new YAML file for more default settings',
                                       required=True)

    # Defaults for computer specific paths in .gitignore
    default_parsers = []
    if os.path.exists('dataset_path.txt'):
        with open('dataset_path.txt', 'r') as file:
            data_dir = file.read()
    else:
        data_dir = os.path.join(os.curdir, 'dataset')
    if os.path.exists('models_path.txt'):
        with open('models_path.txt', 'r') as file:
            model_pardir = file.read()
    else:
        model_pardir = os.path.join(os.curdir, 'models')

    # Defaults from the setups folder
    for setup_name in os.listdir(os.path.join(os.path.curdir, 'setups')):
        name, ext = os.path.splitext(setup_name)
        assert ext == '.yaml', 'Experiment folder should only contain yaml files'
        default_parser = subparsers.add_parser(name)
        with open(os.path.join(os.path.curdir, 'setups', setup_name), 'r') as default_file:
            default_values = yaml.safe_load(default_file)
        parser_add_arguments(default_parser)
        if default_values['model_head'] == 'VQVAE':
            parser_add_vqvae_arguments(default_parser)
        default_parser.set_defaults(name=name, data_dir=data_dir, model_pardir=model_pardir, **default_values)
        default_parsers.append(default_parser)

    args = parser.parse_args()
    if args.model_head != 'Oracle':
        exp_name = [args.name,
                    args.exp,
                    *args.select_classes,
                    'final' if args.final else '']
        args.exp_name = '_'.join(filter(bool, exp_name))
    else:
        args.exp_name = '_'.join(filter(bool, ['Oracle'] + args.select_classes))
    args.m_test = args.m_test if args.m_test else args.input_points
    args.m_training = args.m_test if args.m_test else args.input_points
    args.device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu')
    if args.seed:
        torch.manual_seed = np.random.seed = args.seed
    return args
