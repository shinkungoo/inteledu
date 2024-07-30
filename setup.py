from setuptools import setup, find_packages

setup(
    name='inscd',
    version='1.1.0',
    author='Junhao Shen',
    author_email='shenjh@stu.ecnu.edu.cn',
    license='MIT license',
    packages=find_packages(),
    install_requires=[
        "torch", "tqdm", "numpy>=1.16.5", "scikit-learn", "pandas", "dgl"
    ],
    entry_points={},
)