import sys

from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy
import mc_lib


#extra_compile_args_ = ["-O2"]
extra_compile_args_ = []


ext = Extension("data_simulate", ["data_simulate.pyx"],
                include_dirs = [numpy.get_include(),
                                mc_lib.get_include()],
                language='c++',
                extra_compile_args=extra_compile_args_
                )


setup(ext_modules=cythonize([ext], 
      compiler_directives={'language_level' : "3"}),
      cmdclass = {'build_ext': build_ext})



# OPT="-DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes" python3 setup.py build_ext
# export CFLAGS='-I/usr/lib/python3.7/site-packages/numpy/core/include/'
# HINT: >> python3 setup.py build_ext --inplace