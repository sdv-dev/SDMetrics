#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()

install_requires = [
    'sdv>=0.5.0,<0.6',
    'rdt>=0.2.8.dev0,<0.3',
    'scikit-learn>=0.20,<1',
    'scipy>=1.1.0,<2',
    'numpy>=1.15.4,<2',
    'pandas>=0.21,<2',
    'seaborn>=0.9,<0.11',
    'matplotlib>=2.2.2,<3.2.2',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'parameterized',
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'pytest-rerunfailures>=9.1.1,<10',
    'jupyter>=1.0.0,<2',
    'rundoc>=0.4.3,<0.5',
    'invoke',
]

development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.1',
    'nbsphinx>=0.5.0',
    'Sphinx>=2.4.0,<3.0.0',
    'sphinx_rtd_theme>=0.2.4',
    'autodocsumm>=0.1.13',

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',

    # fix style issues
    'autoflake>=1.2',
    'autopep8>=1.4.3',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1',
    'tox>=2.9.1',
]

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description='Metrics for Synthetic Data Generation Projects',
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    install_package_data=True,
    install_requires=install_requires,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='sdmetrics sdmetrics SDMetrics',
    name='sdmetrics',
    packages=find_packages(include=['sdmetrics', 'sdmetrics.*']),
    python_requires='>=3.6,<3.9',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sdv-dev/SDMetrics',
    version='0.0.4.dev0',
    zip_safe=False,
)
