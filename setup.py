from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='Climate Prediction',
    ext_modules=cythonize("climate2.pyx", annotate=True),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
