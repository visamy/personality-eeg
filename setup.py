from setuptools import setup, find_packages, find_namespace_packages

setup(
    name='src',
    packages=(find_packages() + find_namespace_packages(include=['deepexplain.*'])),
    version='1.0.0',
    description='Big Five personality classification from EEG signals (AMIGOS dataset)',
    author='visamy',
    url='http:/github.com/visamy/personality-eeg',
    license='MIT',
)
