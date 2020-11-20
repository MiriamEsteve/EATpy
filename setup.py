from setuptools import setup

setup(
    name='eat',
    version='0.1',
    description='Efficiency Analysis Trees Technique',
    url='https://doi.org/10.1016/j.eswa.2020.113783',
    author='Miriam Esteve',
    author_email='miriam.estevec@umh.es',
    packages=['eat'],
    install_requires=['numpy', 'pandas', 'graphviz', 'docplex'],
    license='AFL-3.0',
    zip_safe=False
)