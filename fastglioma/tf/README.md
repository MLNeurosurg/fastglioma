## Tensorflow implementation

This directory contains the TensorFlow implementation of the FastGlioma model, for investigational use on the NIO imager. `resnet.py` and `transformer.py` are re-implementations of the PyTorch models in the `models/` directory. Similar to the `inference/` directory, the `feedforward.yaml` file is used to specify the model and inference parameters and the `feedforward.py` script is used to predict on the OpenSRH dataset, starting from the FastGlioma PyTorch checkpoint on HuggingFace. `feedforward.py` has an additional flag, `eval/compare_to_torch`, to directly compare the outputs of the PyTorch and TensorFlow models.

All intermediate outputs and final logits of the TensorFlow implementation satisfy an absolute tolerance of 1e-5 of the original PyTorch implementation.

