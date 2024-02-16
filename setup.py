from setuptools import setup, find_packages

setup(
    name="fastglioma",
    version="0.0",
    packages=find_packages(),
    install_requires=[
        "setuptools>=62.0.0",
        "pip>=22.0.0",
        "pytest",
        "yapf",
        "tqdm",
        "pyyaml",
        "pandas",
        "matplotlib",
        "tifffile",
        "scikit-learn",
        "scikit-image",
        "opencv-python>=4.2.0.34",
        "torch>=1.11",
        "torchvision>=0.11.0",
        "pytorch-lightning",
        "timm",
        "tensorboard" # yapf:disable
    ])