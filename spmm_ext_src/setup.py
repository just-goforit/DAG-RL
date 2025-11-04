from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='spmm_ext',
      ext_modules=[CUDAExtension(name='spmm_ext',
                                 sources=['spmm_ext.cu'],
                                 extra_compile_args={'cxx': ['-j 8', '-O2'],
                                                     'nvcc': ['-O2']})],
      cmdclass={'build_ext': BuildExtension})