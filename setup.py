from setuptools import setup, find_packages

setup(
    name='systemdynamics',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'openpyxl>=3.1.0',
        'seaborn>=0.11.0',
        'xlsxwriter>=3.0.5',
        'scipy',
        'tqdm',
        'sympy',
        'networkx',
        'networkx >= 2.5',
        'ipywidgets',
        'numexpr>=2.8.4',
        'bottleneck>=1.3.6'
    ],
    entry_points={
        'console_scripts': [
            # Define any scripts or entry points for command-line use
        ],
    },
)