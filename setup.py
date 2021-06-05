import glob
import json

from setuptools import setup, find_packages

from binance_prediction import __version__


def get_packages_from_pipfile_lock(path: str):
    with open(path, "r") as read_file:
        data = json.load(read_file)
        packages_list = [f"{item[0]}{item[1]['version']}" for item in data['default'].items()]
        packages = '\n'.join(packages_list)
        return packages


setup(
    name='binance_prediction',
    version=__version__,
    packages=(find_packages()),
    description='{package description}',
    long_description=open('README.md').read(),
    url='https://github.com/Pepslee/binance_prediction',
    install_requires=get_packages_from_pipfile_lock('Pipfile.lock'),
    scripts=glob.glob('bin/*.py'),
    test_suite='tests',
    include_package_data=True
)
