"""Build ShapBPT with SAM guidance (this repository's fork of github.com/amparore/shap_bpt).

The version lives in shap_bpt/_version.py (history of what each version added is listed there); check an
installed build with:
    python -c "import shap_bpt, pprint; pprint.pprint(shap_bpt.build_info())"

The Cython extension shap_bpt/bpt.pyx is compiled per CPU type and Python version
(bpt.cpython-312-aarch64-linux-gnu.so on EPITO, ...-x86_64-linux-gnu.so on x86 nodes), so rebuild it on each:
    python -m pip install --no-deps --no-build-isolation --force-reinstall -e .
Compiling needs Python.h: on HPC4AI mercurio the login node has no Python headers, so build the x86 extension on
an x86 GPU node (e.g. bash src/hpc/gpu.sh v100 -- ...) and the ARM one on EPITO.
"""
import re
from pathlib import Path

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

VERSION = re.search(r"__version__ = '([^']+)'",
                    (Path(__file__).parent / 'shap_bpt' / '_version.py').read_text()).group(1)

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
    version=VERSION,
    description='ShapBPT with SAM guidance: Owen-value image explanations on SAM-guided Binary Partition Trees',
    long_description=(
        'ShapBPT explains image models with Owen values estimated over a Binary Partition Tree. This version '
        'makes the tree object-aware: Segment Anything (SAM-1/SAM-2) masks guide the BPT merge order '
        '(BPT+SAM_c). Paper: "ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees", '
        'AAAI 2026. Version history: shap_bpt/_version.py.'),
    long_description_content_type='text/plain',
    # Authors as in CITATION.cff; contact and maintainer: Muhammad Rashid
    author='Muhammad Rashid, Elvio G. Amparore, Enrico Ferrari, Damiano Verda',
    author_email='muhammadrashid8045@gmail.com',
    maintainer='Muhammad Rashid',
    maintainer_email='muhammadrashid8045@gmail.com',
    url='https://github.com/amparore/shap_bpt',
    project_urls={'Source': 'https://github.com/amparore/shap_bpt',
                  'Citation': 'https://github.com/amparore/shap_bpt/blob/main/CITATION.cff'},
    license='MIT',
    keywords=['explainable AI', 'XAI', 'Shapley values', 'Owen values', 'binary partition trees',
              'feature attribution', 'segment anything', 'SAM'],
    classifiers=['Programming Language :: Python :: 3',
                 'Programming Language :: Cython',
                 'Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Image Processing'],
    python_requires='>=3.10',
    packages=['shap_bpt'],
    ext_modules=ext_modules,
    install_requires=['Cython',
                      'numpy',
                      'matplotlib',
                      'tqdm',
                      'scikit-image'],
)