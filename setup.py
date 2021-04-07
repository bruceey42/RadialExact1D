# -*- coding: utf-8 -*-
import setuptools
 
setuptools.setup(
    name="thermpy1d",
    version="0.01",
    description="1d thermal tool",
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