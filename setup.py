from setuptools import setup, find_packages

setup(
    name='systemdynamics',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'openpyxl>=3.1.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'scipy',
        'networkx>=2.5',
        'jax',
        'diffrax',
        'tqdm',
        'ipywidgets',
        'tabulate>=0.8.9',
    ],
)
