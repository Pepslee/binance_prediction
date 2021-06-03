import glob

from setuptools import setup, find_packages

from binance_prediction import __version__

setup(
    name='binance_prediction',
    version=__version__,
    packages=(find_packages()),
    description='{package description}',
    long_description=open('README.md').read(),
    url='https://github.com/Pepslee/binance_prediction',
    install_requires=open('requirements.txt').read(),
    scripts=glob.glob('bin/[!_]*'),
    test_suite='tests',
    include_package_data=True
)
