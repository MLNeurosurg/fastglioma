from setuptools import setup, find_packages

setup(
    name="fastglioma",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "setuptools",
        "pip",
        "pytest",
        "yapf",
        "tqdm",
        "pyyaml",
        "pytest==8.3.2",
        "pandas==1.5.3",
        "numpy==1.24.4",
        "matplotlib==3.6.3",
        "tifffile==2020.10.1",
        "scikit-learn==1.4.1.post1",
        "scikit-image",
        "opencv-python==3.4.18.65",
        "torch==1.13.0",
        "torchvision==0.14.0",
        "pytorch-lightning==1.8.4",
        "huggingface-hub==0.24.6",
        "timm",
        "tensorboard" # yapf:disable
    ])
