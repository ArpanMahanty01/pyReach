from setuptools import setup, find_packages

setup(
    name='pyReach',
    version='0.1.0',
    author='Arpan Mahanty',
    description='A reachability analysis toolbox',
    packages=find_packages('pyReach','pyReach.*'),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
    ],
    python_requires='>=3.7',
)