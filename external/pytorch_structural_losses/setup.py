from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Python interface
setup(
    name='structural_losses',
    version='0.1.0',
    install_requires=['torch'],
    packages=['structural_losses'],
    package_data={'structural_losses': ['py.typed', '*.pyi']},
    ext_modules=[
        CUDAExtension(
            name='structural_losses.structural_losses_backend',
            sources=[
                'src/approxmatch.cu',
                'src/nndistance.cu',
                'src/structural_loss.cpp',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
