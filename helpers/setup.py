from setuptools import setup, find_packages


setup(
    name='helpers',
    version='0.0.1',
    license='MIT',
    author='Jacob Gerlach',
    author_email='jwgerlach00@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.7',
)
