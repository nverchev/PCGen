from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    install_requires=['torch'],
    packages=['emd'],
    ext_modules=[
        CUDAExtension(
            name='emd_backend',
            sources=[
                'src/emd.cpp',
                'src/emd_cuda.cu',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
