import os
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the game library")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,]

        cfg = 'Debug' if self.debug else 'Release'
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args = ['--config', cfg, '--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--config', 'Release', '--', '-j2', 'VERBOSE=1'])

        print()  # Add an empty line for cleaner output

setup(
    name="gym_battlesnake_pytorch",
    version="0.0.1",
    author="Bjarni Haukur",
    author_email="bjarnihaukur11@gmail.com",
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/BjarniHaukur/gym-battlesnake-pytorch",
    packages=["gym_battlesnake_pytorch"],
    install_requires=[
        'gym',
        'numpy',
        'stable-baselines3',
    ],
    ext_modules=[CMakeExtension('gym_battlesnake/gym_battlesnake')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
