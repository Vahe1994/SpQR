from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class BuildWithCompilerDetection(BuildExtension):
    def build_extensions(self):
        super().build_extensions()


setup(
    name="spqr",
    version="0.1.1",
    author="elvircrn",
    author_email="elvircrn@gmail.com",
    description="SPQR",
    install_requires=["numpy", "torch"],
    packages=["spqr"],
    ext_modules=[
        CUDAExtension(
            name="spqr_cuda",
            sources=["spqr/spqr/spqr_cuda.cpp", "spqr/spqr/spqr_cuda_kernel.cu", "spqr/spqr/torch_bench.cu"],
            extra_compile_args={"cxx": ["-Wall", "-O3"], "nvcc": ["-O3", "-std=c++17", "-lineinfo", "-arch=native"]},
        )
    ],
    cmdclass={"build_ext": BuildWithCompilerDetection},
)
