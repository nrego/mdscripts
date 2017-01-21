from distutils.core import setup
from distutils.extension import Extension


import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()
from Cython.Distutils import build_ext

ext_modules = [Extension("rhoutils.utils", ["rhoutils/utils.pyx"],
            include_dirs=[numpy_include],
            extra_compile_args=["-O3","-ffast-math"]),
                Extension("boxutils.utils", ["boxutils/utils.pyx"],
                            include_dirs=[numpy_include],
                            extra_compile_args=["-O3","-ffast-math"])]
setup(cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)
