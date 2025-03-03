from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

version = {}
with open("visionlab/datasets/version.py") as fp:
    exec(fp.read(), version)

setup(
    name="visionlab-datasets",
    version=version['__version__'],
    author="George Alvarez",
    author_email="alvarez@wjh.harvard.edu",
    description="Datasets for cog-neuro-ai research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harvard-visionlab/datasets",
    packages=find_namespace_packages(include=["visionlab.*"]),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'download_rawdata=visionlab.datasets.cli.download_rawdata:main',
            'generate_lightning_dataset=visionlab.datasets.cli.generate_lightning_dataset:main',
            'check_dataset=visionlab.datasets.cli.check_dataset:main',
            'sync_dataset=visionlab.datasets.cli.sync_dataset:main',
            'generate_ffcv_dataset=visionlab.datasets.cli.generate_ffcv_dataset:main',
        ],
    },
)