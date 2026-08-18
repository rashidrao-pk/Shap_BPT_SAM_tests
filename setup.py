from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

import platform
is_windows = (platform.system()=='Windows')
is_mingw = is_windows and ('--compiler=mingw32' in sys.argv)

extra_compile_args=['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION', 
                    '-Wunreachable-code']

if is_mingw:
    extra_compile_args.append('-DMS_WIN64')


ext_modules = cythonize([
    Extension("shap_bpt.bpt", ["shap_bpt/bpt.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=extra_compile_args
    ),
])

setup(
    name='shap_bpt',
    version='1.0',
    packages=['shap_bpt'],
    ext_modules=ext_modules,
    install_requires=['Cython',
                      'numpy',
                      'matplotlib',
                      'tqdm',
                      'scikit-image'],
)