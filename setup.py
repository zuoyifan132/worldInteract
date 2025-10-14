"""
Setup configuration for worldInteract package
"""

from setuptools import setup, find_packages

setup(
    name="worldInteract",
    version="0.1.0",
    description="World Interaction Environment Generator",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "networkx>=2.6",
        "numpy>=1.20",
        "loguru>=0.5",
        "tqdm>=4.60",
        "pyyaml>=5.4",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

