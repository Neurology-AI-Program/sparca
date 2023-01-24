from setuptools import setup, setuptools
from sparca._version import __version__

setup(
    name = 'sparca.py',
    packages = setuptools.find_packages(),
    author = 'Leland Barnard',
    version = __version__
)