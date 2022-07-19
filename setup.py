from setuptools import find_packages, setup

setup(
    name="pointscatter",
    version="0.1",
    install_requires=["torch>=1.7", "mmsegmentation>=0.19.0"],
    packages=find_packages(exclude=("configs",)),
    python_requires=">=3.6",
)
