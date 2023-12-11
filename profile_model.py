import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from src.options import parse_process_args_and_set_seed
from src.model import get_model


@torch.inference_mode()
def profile_model():
    args = parse_process_args_and_set_seed(task="profile", description='Estimate model computational cost')
    model = get_model(**vars(args))
    model = model.eval()
    model.to(args.device)
    if args.model_head == 'VQVAE':
        model.forward = model.random_sampling  # Overwrites forward method to profile random generation
        dummy_input = [args.batch_size]
    else:
        dummy_input = [torch.ones(args.batch_size, args.input_points, 3, device=args.device),
                       torch.zeros(args.batch_size, args.input_points, args.k, device=args.device, dtype=torch.long)]

    get_model_profile(model=model,
                      args=dummy_input,
                      print_profile=True,
                      detailed=True,
                      module_depth=2,
                      top_modules=2,
                      warm_up=10,
                      as_string=True,
                      output_file=None,
                      ignore_modules=None)


if __name__ == '__main__':
    profile_model()
