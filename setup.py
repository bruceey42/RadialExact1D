# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="thermpy1d",
    version="0.01",
    description="1d thermal tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Bruce Edwards",
    author_email="bruce.edwards@ukaea.uk",
    url="https://github.com/bruceey42/ThermPy1D",
    packages=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib"
    ],
    python_requires=">=3.6",
)