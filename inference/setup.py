from setuptools import setup
from torch.utils import cpp_extension


class build_ext_with_compiler_detection(cpp_extension.BuildExtension):
    def build_extensions(self):
        # self.compiler.linker_so[0] = 'mold'
        super().build_extensions()

setup(
    name='spqr',
    version='0.1.1',
    author='elvircrn',
    author_email='elvircrn@gmail.com',
    description='SPQR',
    install_requires=['numpy', 'torch'],
    packages=['inference'],
    ext_modules=[
        cpp_extension.CUDAExtension(
            'spqr_cuda',
            [
                'inference/spqr/spqr_cuda.cpp',
                'inference/spqr/spqr_cuda_kernel.cu',
                'inference/spqr/torch_bench.cu'
            ],
            extra_compile_args={'cxx': [
                '-fuse-ld=mold',
                '-Wall',
            ],
                'nvcc': [
                    # https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py#L1050C13-L1050C17
                    '-O3',
                    '-std=c++17',
                    '-lineinfo',
                    '-arch=native'
                ]}
        )
    ],
    cmdclass={'build_ext': build_ext_with_compiler_detection},
)
