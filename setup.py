import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(
        "pjsk_tts/monotonic_align/core.pyx",
        language_level="3",
    ),
    include_dirs=[numpy.get_include()],
)
