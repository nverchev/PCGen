from deepspeed.profiling.flops_profiler import get_model_profile
from main import main


def profile_model():
    model, dummy_input = main(task='return model for profiling')
    flops, macs, params = get_model_profile(model=model,
                                            args=dummy_input,
                                            print_profile=True,
                                            detailed=False,
                                            module_depth=2,
                                            top_modules=2,
                                            warm_up=10,
                                            as_string=True,
                                            output_file=None,
                                            ignore_modules=None)


if __name__ == '__main__':
    profile_model()
