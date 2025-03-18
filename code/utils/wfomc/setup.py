from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

cython_modules = [
    Extension(
        name="cython_modules.matrix_utils",
        sources=["cython_modules/matrix_utils.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(cython_modules),
)