from setuptools import setup, find_packages
import os, numpy, platform, subprocess
from Cython.Distutils import build_ext
from distutils.extension import Extension

if "CUDA_PATH" in os.environ:
    CUDA_PATH = os.environ["CUDA_PATH"]
else:
    CUDA_PATH = "/usr/local/cuda"


abspath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multinomial_mix")
cpu_wrappers = Extension("core_cpu_func_wrappers",
        sources = [os.path.join(abspath, "core_cpu_func_wrappers.pyx"),
            os.path.join(abspath, "prob_calcs.c")],
        language="c",
        include_dirs=[numpy.get_include(), abspath],
        library_dirs=[abspath],
        extra_compile_args=["-O3"])

cpu_wrappers.cython_directives = {"language_level":"3"}

#cuda_wrappers = Extension("core_cuda_func_wrappers",
#        sources = ["core_cuda_func_wrappers.pyx"],
#        language="c++",
#        libraries = ["gpu_prob_calcs", "cudart_static", "cuda"],
#        include_dirs=[numpy.get_include(),
#            os.path.dirname(os.path.abspath((__file__)))],
#        library_dirs=[os.path.dirname(os.path.abspath((__file__))),
#                    os.path.join(CUDA_PATH, "lib64")],
#        extra_link_args = ["-lrt"]
#        )

#cuda_wrappers.cython_directives = {"language_level":"3"}



setup(
        name="multinomial_mix",
        version="0.0.0.1",
        packages=find_packages(),
        cmdclass = {"build_ext": build_ext},
        author="Jonathan Parkinson",
        author_email="jlparkinson1@gmail.com",
        ext_modules = [cpu_wrappers],
        package_data={"": ["*.h", "*.c", "*.cu",
            "*.pyx", "*.sh"]}
    )

#os.remove("core_cuda_func_wrappers.cpp")
#os.remove("core_cpu_func_wrappers.c")
#os.remove("libprob_calcs.a")
#os.remove("prob_calcs.o")
