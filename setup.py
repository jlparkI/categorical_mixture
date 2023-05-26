"""Setup for the """
import os
from setuptools import setup, find_packages
import numpy
from Cython.Distutils import build_ext
#Irrespective of what your linter says, do not
#move this import above setuptools; this import
#monkey patches setuptools, and -- remarkably --
#reversing the order will lead to an error.
from distutils.extension import Extension


def get_version(start_fpath):
    """Retrieves the version number."""
    os.chdir(os.path.join(start_fpath, "multinoulli_mixture"))
    with open("__init__.py", "r") as fhandle:
        version_line = [l for l in fhandle.readlines() if
                l.startswith("__version__")]
        version = version_line[0].split("=")[1].strip().replace('"', "")
    os.chdir(start_fpath)
    return version



setup_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)))
readme_path = os.path.join(setup_fpath, "README.md")

with open(readme_path, "r") as fhandle:
    long_description = "".join(fhandle.readlines())

extension_path = os.path.join(setup_fpath, "multinoulli_mixture",
        "core_cpu_funcs")
cpu_wrappers = Extension("core_cpu_func_wrappers",
        sources = [os.path.join(extension_path, "core_cpu_func_wrappers.pyx"),
            os.path.join(extension_path, "responsibility_calcs.cpp"),
            os.path.join(extension_path, "weighted_counts.cpp")],
        language="c++",
        include_dirs=[numpy.get_include(), extension_path],
        extra_compile_args=["-O3"])

cpu_wrappers.cython_directives = {"language_level":"3"}

setup(
        name="multinoulli_mix",
        version=get_version(setup_fpath),
        packages=find_packages(),
        cmdclass = {"build_ext": build_ext},
        author="Jonathan Parkinson",
        author_email="jlparkinson1@gmail.com",
        description = "Fitting a mixture of multinoullis model to sequence data",
        long_description = long_description,
        long_description_content_type = "text/markdown",
        install_requires = ["numpy>=1.10", "scipy>=1.7.0",
            "cython>=0.10"],
        ext_modules = [cpu_wrappers],
        package_data={"": ["*.h", "*.c", "*.cpp",
            "*.pyx", "*.sh"]}
    )

os.chdir(extension_path)
os.remove("core_cpu_func_wrappers.cpp")
