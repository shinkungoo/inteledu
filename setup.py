from setuptools import setup, find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8<5.0.0',
    'flake8<5.0.0'
]

setup(
    name='inscd',
    version='1.1.0',
    extras_require={
        'test': test_deps,
    },
    packages=find_packages(),
    install_requires=[
        "torch", "tqdm", "numpy>=1.16.5", "scikit-learn", "pandas", "dgl"
    ],
    entry_points={},
)