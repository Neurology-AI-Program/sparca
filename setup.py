from setuptools import setup, setuptools
from sparca._version import __version__

setup(
    name = 'sparca.py',
    packages = setuptools.find_packages(),
    author = 'Leland Barnard',
    version = __version__,
    include_package_data=True,
    package_data = {'sparca': ['test_data/*']}
)