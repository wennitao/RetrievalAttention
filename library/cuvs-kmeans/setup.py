import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

src_dir = "balanced_kmeans/src"

ext_modules = [
    CUDAExtension(
        'balanced_kmeans.BalancedKmeans',
        sources=[f'{src_dir}/balanced_kmeans.cu'],
        include_dirs=['/usr/local/cuda-12.8/include', '/home/nvidia/user/conda/envs/retroinfer/include'],
        library_dirs=['/usr/local/lib', '/home/nvidia/user/conda/envs/retroinfer/lib'],
        extra_compile_args={'cxx': ['-O3', '-std=c++17'],
                          'nvcc': ['-O3', '-std=c++17', '--expt-relaxed-constexpr', '--extended-lambda', '-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE']},
        extra_link_args=['-lcuda', '-lcudart', '-lcuvs',
                        '-Wl,-rpath,/home/nvidia/user/conda/envs/retroinfer/lib',
                        '-Wl,-rpath,/home/nvidia/user/conda/envs/retroinfer/lib/python3.10/site-packages/torch/lib'],
    ),
]


setup(
    name='balanced_kmeans',
    version='0.1',
    packages=['balanced_kmeans'],
    description='Balanced KMeans clustering',
    long_description='A collection of CUDA and C++ extensions for Balanced KMeans clustering.',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=['pybind11', 'torch==2.5.1'],
    python_requires='>=3.10',
)