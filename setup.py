from setuptools import setup, find_packages

from binance_prediction import __version__
from binance_prediction import __package_name__


setup(
    name=__package_name__,
    version=__version__,
    packages=(find_packages()),
    description='{package description}',
    long_description=open('README.md').read(),
    url='https://github.com/Pepslee/binance_prediction',
    install_requires=open('requirements.txt').read(),
    scripts=['binance_prediction/train.py', ],
    test_suite='tests',
    include_package_data=True
)
