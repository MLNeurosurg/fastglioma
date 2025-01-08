"""ResNet backbone in TensorFlow.

Matches PyTorch implementation exactly.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Optional, Any

class PyTorchBatchNorm(keras.layers.Layer):
    """BatchNorm that exactly matches PyTorch's BatchNorm2d"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, 
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.gamma = self.add_weight(
                'gamma',
                shape=[num_features],
                initializer='ones',
                trainable=True)
            self.beta = self.add_weight(
                'beta',
                shape=[num_features],
                initializer='zeros',
                trainable=True)
        else:
            self.gamma = None
            self.beta = None
        
        if self.track_running_stats:
            self.moving_mean = self.add_weight(
                'moving_mean',
                shape=[num_features],
                initializer='zeros',
                trainable=False)
            self.moving_var = self.add_weight(
                'moving_variance',
                shape=[num_features],
                initializer='ones',
                trainable=False)
            self.num_batches_tracked = self.add_weight(
                'num_batches_tracked',
                shape=[],
                initializer='zeros',
                trainable=False,
                dtype=tf.int64)
        else:
            self.moving_mean = None
            self.moving_var = None
            self.num_batches_tracked = None
    
    def call(self, x, training=False):
        # Check input dimensions
        if len(x.shape) != 4:
            raise ValueError(f"expected 4D input (got {len(x.shape)}D input)")
            
        if training or not self.track_running_stats:
        # if True:
            axes = [0, 1, 2]  # Calculate over N,H,W dimensions
            batch_mean = tf.reduce_mean(x, axes, keepdims=True)
            
            # Calculate variance exactly as PyTorch does
            batch_var = tf.reduce_mean(
                tf.square(x - batch_mean), 
                axes,
                keepdims=True
            )
            
            # if training and self.track_running_stats:
            #     # Update running stats
            #     mean_update = tf.reduce_mean(batch_mean, axis=[0, 1, 2])
                
            #     # Use unbiased variance for running stats (torch.var(input, unbiased=True))
            #     var_update = tf.reduce_sum(
            #         tf.square(centered_x), axes) / (n - 1)  # unbiased variance
            #     var_update = tf.reduce_mean(var_update, axis=[0, 1, 2])
                
            #     # Update using momentum: new = (1-momentum) * old + momentum * current
            #     self.moving_mean.assign(
            #         (1 - self.momentum) * self.moving_mean + 
            #         self.momentum * mean_update)
            #     self.moving_var.assign(
            #         (1 - self.momentum) * self.moving_var + 
            #         self.momentum * var_update)
                
            #     self.num_batches_tracked.assign_add(1)
            
            mean = batch_mean
            var = batch_var
        else:
            mean = self.moving_mean[None, None, None, :]
            var = self.moving_var[None, None, None, :]
        
        # Normalize: (x - E[x]) / sqrt(Var[x] + eps)
        x_norm = (x - mean) / tf.sqrt(var + self.eps)
        
        # Scale and shift
        if self.affine:
            gamma = self.gamma[None, None, None, :]
            beta = self.beta[None, None, None, :]
            return gamma * x_norm + beta
        return x_norm

# Helper for exact PyTorch initialization
def pytorch_conv_init():
    """Returns initializer that exactly matches PyTorch's default Conv2d initialization.
    
    PyTorch uses Kaiming initialization by default:
    - mode='fan_out'
    - nonlinearity='relu'
    - distribution='normal'
    """
    def init(shape, dtype=None):
        # For Conv2d, shape is (kernel_h, kernel_w, in_channels, out_channels)
        if len(shape) != 4:
            raise ValueError(f"Expected 4D shape for Conv2d weights, got {len(shape)}D")
            
        # Calculate fan_out for Conv2d
        # fan_out = kernel_size * kernel_size * out_channels
        kernel_h, kernel_w, _, out_channels = shape
        fan_out = kernel_h * kernel_w * out_channels
        
        # Calculate gain for ReLU (sqrt(2))
        gain = np.sqrt(2.0)
        
        # Calculate std according to PyTorch's formula
        std = gain / np.sqrt(fan_out)
        
        # Generate normal distribution with calculated std
        weights = tf.random.normal(shape, mean=0.0, stddev=std, dtype=dtype)
        return weights
    
    return init

def conv3x3(filters: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1) -> keras.layers.Layer:
    """3x3 convolution with padding that matches PyTorch's Conv2d exactly"""
    # Explicit padding to match PyTorch exactly
    padding_layer = keras.layers.ZeroPadding2D(padding=dilation)
    conv_layer = keras.layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=stride,
        padding='valid',  # Use explicit padding instead of 'same'
        groups=groups,
        use_bias=False,
        dilation_rate=dilation,
        kernel_initializer=pytorch_conv_init(),
        data_format='channels_last'
    )
    return keras.Sequential([padding_layer, conv_layer])

def conv1x1(filters: int, stride: int = 1, input_channels: Optional[int] = None) -> keras.layers.Layer:
    """1x1 convolution that matches PyTorch's Conv2d exactly"""
    if input_channels is not None:
        # Create a custom initializer that knows about input channels
        def custom_init(shape, dtype=None):
            # shape will be (1, 1, input_channels, filters)
            assert shape[0] == 1 and shape[1] == 1, "Expected 1x1 kernel"
            assert shape[2] == input_channels, f"Expected {input_channels} input channels, got {shape[2]}"
            assert shape[3] == filters, f"Expected {filters} filters, got {shape[3]}"
            # Use the same initialization as PyTorch's default
            return pytorch_conv_init()(shape, dtype)
        
        kernel_initializer = custom_init
    else:
        kernel_initializer = pytorch_conv_init()

    return keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(stride, stride),
        padding='valid',
        use_bias=False,
        kernel_initializer=kernel_initializer,
        data_format='channels_last'
    )
 
class BasicBlock(keras.layers.Layer):
    expansion = 1

    def __init__(self, filters, stride=1, downsample=None, groups=1, 
                 base_width=64, dilation=1, **kwargs):
        super().__init__(**kwargs)
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(filters, stride)
        self.bn1 = PyTorchBatchNorm(filters)
        self.relu = keras.layers.ReLU()
        self.conv2 = conv3x3(filters)
        self.bn2 = PyTorchBatchNorm(filters)
        self.downsample = downsample
        self.stride = stride

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out = tf.add(out, identity)
        out = self.relu(out)

        return out

class Bottleneck(keras.layers.Layer):
    expansion = 4

    def __init__(self, filters, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, **kwargs):
        super().__init__(**kwargs)
        width = int(filters * (base_width / 64.0)) * groups
        
        self.conv1 = conv1x1(width)
        self.bn1 = PyTorchBatchNorm(width)
        self.conv2 = conv3x3(width, stride, groups, dilation)
        self.bn2 = PyTorchBatchNorm(width)
        self.conv3 = conv1x1(filters * self.expansion)
        self.bn3 = PyTorchBatchNorm(filters * self.expansion)
        self.relu = keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out = tf.math.add(out, identity)
        out = self.relu(out)

        return out

class ResNetBackbone(keras.Model):
    def __init__(self, 
                 num_channel_in: int,
                 in_planes: int,
                 layer_planes: List[int],
                 block: keras.layers.Layer,
                 layers: List[int],
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.inplanes = in_planes
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
            
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial layers with exact same initialization as PyTorch
        padding_layer = keras.layers.ZeroPadding2D(padding=3)  # Explicit padding of 3 pixels
        conv_layer = keras.layers.Conv2D(
            self.inplanes, 
            kernel_size=7,
            strides=2,
            padding='valid',  # Use explicit padding
            use_bias=False,
            kernel_initializer=pytorch_conv_init(),
            data_format='channels_last'
        )
        self.conv1 = keras.Sequential([padding_layer, conv_layer])
        self.bn1 = PyTorchBatchNorm(self.inplanes)
        self.relu = keras.layers.ReLU()
        
        # Match PyTorch's maxpool exactly: kernel_size=3, stride=2, padding=1
        self.maxpool = keras.layers.ZeroPadding2D(padding=1)
        self.maxpool2 = keras.layers.MaxPool2D(
            pool_size=3, 
            strides=2, 
            padding='valid',
            dtype=tf.float32
        )
        
        # ResNet layers
        self.layer1 = self._make_layer(block, layer_planes[0], layers[0])
        self.layer2 = self._make_layer(block, layer_planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, layer_planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, layer_planes[3], layers[3], stride=2)
        
        # Match PyTorch's adaptive_avg_pool2d with output_size=1
        self.avgpool = keras.layers.AveragePooling2D(
            pool_size=(1, 1),  # Default value, will be ignored
            strides=(1, 1),    # Default value, will be ignored
            padding='valid',
            data_format='channels_last'
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential([
                conv1x1(planes * block.expansion, stride, input_channels=self.inplanes),
                PyTorchBatchNorm(planes * block.expansion)
            ])
        
        layers = []
        layers.append(
            block(planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(planes, groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))
        
        return keras.Sequential(layers)

    def call(self, x, training=False):
        x = tf.cast(x, tf.float32)
        
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.maxpool2(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        # Compute pool_size dynamically to match PyTorch's adaptive_avg_pool2d
        h, w = x.shape[1:3]
        x = tf.nn.avg_pool2d(
            x,
            ksize=[1, h, w, 1],
            strides=[1, h, w, 1],
            padding='VALID',
            data_format='NHWC'
        )  # Output shape: (batch, 1, 1, channels)
        print("After avgpool:\n")
        print(tf.reduce_mean(x))
        print(tf.reduce_min(x))
        print(tf.reduce_max(x))

        return x

def resnet_backbone(arch: str = 'resnet50',
                   num_channel_in: int = 3,
                   in_planes: int = 64,
                   layer_planes: List[int] = [64, 128, 256, 512],
                   **kwargs: Any) -> ResNetBackbone:
    """Creates a resnet backbone."""
    blocks = {
        'resnet18': BasicBlock,
        'resnet34': BasicBlock,
        'resnet50': Bottleneck,
        'resnet101': Bottleneck,
        'resnet152': Bottleneck,
    }

    layers = {
        'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3],
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        'resnet152': [3, 8, 36, 3],
    }

    return ResNetBackbone(
        num_channel_in=num_channel_in,
        in_planes=in_planes,
        layer_planes=layer_planes,
        block=blocks[arch],
        layers=layers[arch],
        **kwargs)

def convert_resnet_weights(pytorch_model, tf_model):
    """Convert PyTorch ResNet weights to TensorFlow and verify the conversion"""
    # Initialize TF model
    batch_size = 1
    dummy_input = tf.zeros((batch_size, 224, 224, 3), dtype=tf.float32)
    _ = tf_model(dummy_input, training=False)
    
    state_dict = pytorch_model.state_dict()
    tf_weights = []
    
    # Initial conv + bn
    print("\nVerifying initial layers:")
    print("Conv1:")
    tf_weights.append(state_dict['conv1.weight'].numpy().transpose(2, 3, 1, 0))  # Keep TF format for setting weights
    
    # Initial BatchNorm
    print("- BN1")
    tf_weights.append(state_dict['bn1.weight'].numpy())  # gamma
    tf_weights.append(state_dict['bn1.bias'].numpy())    # beta
    tf_weights.append(state_dict['bn1.running_mean'].numpy())  # moving_mean
    tf_weights.append(state_dict['bn1.running_var'].numpy())   # moving_var
    tf_weights.append(state_dict['bn1.num_batches_tracked'].numpy())  # num_batches_tracked

    # ResNet layers
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    for layer_name in layer_names:
        layer = getattr(pytorch_model, layer_name)
        for i in range(len(layer)):
            block_prefix = f'{layer_name}.{i}'
            print(f"\nLoading {block_prefix}:")
            
            # Conv1 + BN1
            print("- Conv1")
            tf_weights.append(state_dict[f'{block_prefix}.conv1.weight'].numpy().transpose(2, 3, 1, 0))
            print("- BN1")
            tf_weights.append(state_dict[f'{block_prefix}.bn1.weight'].numpy())  # gamma
            tf_weights.append(state_dict[f'{block_prefix}.bn1.bias'].numpy())    # beta
            
            # Conv2 + BN2
            print("- Conv2")
            tf_weights.append(state_dict[f'{block_prefix}.conv2.weight'].numpy().transpose(2, 3, 1, 0))
            print("- BN2")
            tf_weights.append(state_dict[f'{block_prefix}.bn2.weight'].numpy())  # gamma
            tf_weights.append(state_dict[f'{block_prefix}.bn2.bias'].numpy())    # beta
            
            # Downsample (if exists)
            if f'{block_prefix}.downsample.0.weight' in state_dict:
                print("- Downsample")
                tf_weights.append(state_dict[f'{block_prefix}.downsample.0.weight'].numpy().transpose(2, 3, 1, 0))
                print("- Downsample BN")
                tf_weights.append(state_dict[f'{block_prefix}.downsample.1.weight'].numpy())  # gamma
                tf_weights.append(state_dict[f'{block_prefix}.downsample.1.bias'].numpy())    # beta
            
            # Add all moving means and variances at the end of each block
            print("- Adding moving statistics")
            tf_weights.append(state_dict[f'{block_prefix}.bn1.running_mean'].numpy())
            tf_weights.append(state_dict[f'{block_prefix}.bn1.running_var'].numpy())
            tf_weights.append(state_dict[f'{block_prefix}.bn1.num_batches_tracked'].numpy())
            tf_weights.append(state_dict[f'{block_prefix}.bn2.running_mean'].numpy())
            tf_weights.append(state_dict[f'{block_prefix}.bn2.running_var'].numpy())
            tf_weights.append(state_dict[f'{block_prefix}.bn2.num_batches_tracked'].numpy())
            if f'{block_prefix}.downsample.0.weight' in state_dict:
                tf_weights.append(state_dict[f'{block_prefix}.downsample.1.running_mean'].numpy())
                tf_weights.append(state_dict[f'{block_prefix}.downsample.1.running_var'].numpy())
                tf_weights.append(state_dict[f'{block_prefix}.downsample.1.num_batches_tracked'].numpy())
    # Debug print for shapes
    print("\nVerifying weight shapes:")
    tf_weights_iter = iter(tf_model.weights)
    for i, pytorch_weight in enumerate(tf_weights):
        tf_weight = next(tf_weights_iter)
        print(f"{i}: TF {tf_weight.name} shape {tf_weight.shape} <- PyTorch shape {pytorch_weight.shape}")
        try:
            assert tf_weight.shape == pytorch_weight.shape, f"Shape mismatch at index {i}"
        except AssertionError as e:
            print(f"Error at weight {i}:")
            print(f"TF weight name: {tf_weight.name}")
            print(f"TF shape: {tf_weight.shape}")
            print(f"PyTorch shape: {pytorch_weight.shape}")
            raise e
    
    # Set the weights
    tf_model.set_weights(tf_weights)
    
    return tf_model

if __name__ == "__main__":
    import torch
    from fastglioma.models.resnet import resnet_backbone as torch_resnet_backbone
    
    torch.backends.cudnn.deterministic = True

    import os
    # Make TensorFlow deterministic
    tf.random.set_seed(0)  # or whatever seed you're using
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new in TF 2.x
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Test different architectures
    for arch in ['resnet34']:
        print(f"\nTesting {arch}...")
        
        # Create models
        pytorch_model = torch_resnet_backbone(arch, num_channel_in=3)
        pytorch_model.eval()
        tf_model = resnet_backbone(arch, num_channel_in=3)
        
        # Initialize TF model
        dummy_input = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        _ = tf_model(dummy_input)
        
        # Convert weights
        tf_model = convert_resnet_weights(pytorch_model, tf_model)
        
        # Test with simple input
        dummy_input_np = np.ones((1, 224, 224, 3), dtype=np.float32)
        
        # PyTorch expects (B, C, H, W)
        dummy_input_torch = torch.from_numpy(dummy_input_np.transpose(0, 3, 1, 2))
        
        # TensorFlow expects (B, H, W, C)
        dummy_input_tf = tf.convert_to_tensor(dummy_input_np)
        
        # Get outputs and intermediate activations
        with torch.no_grad():
            # PyTorch forward pass with hooks to capture intermediate activations
            pytorch_activations = {}
            def hook_fn(name):
                def hook(module, input, output):
                    pytorch_activations[name] = output.numpy()
                return hook

            pytorch_model.conv1.register_forward_hook(hook_fn('conv1'))
            pytorch_model.bn1.register_forward_hook(hook_fn('bn1'))
            pytorch_model.avgpool.register_forward_hook(hook_fn('avgpool'))
            
            output_torch = pytorch_model(dummy_input_torch)
            
            # Print PyTorch stats
            print("\nPyTorch activations:")

            print("After conv1:")
            conv1_torch = pytorch_activations['conv1']
            print(f"Mean: {np.mean(conv1_torch):.6f}")
            print(f"Var: {np.var(conv1_torch):.6f}")
            print(f"Min: {np.min(conv1_torch):.6f}")
            print(f"Max: {np.max(conv1_torch):.6f}")

            print("After bn1:")
            bn1_torch = pytorch_activations['bn1']
            print(f"Mean: {np.mean(bn1_torch):.6f}")
            print(f"Var: {np.var(bn1_torch):.6f}")
            print(f"Min: {np.min(bn1_torch):.6f}")
            print(f"Max: {np.max(bn1_torch):.6f}")

            print("After avgpool:")
            avgpool_torch = pytorch_activations['avgpool']
            print(f"Mean: {np.mean(avgpool_torch):.6f}")
            print(f"Var: {np.var(avgpool_torch):.6f}")
            print(f"Min: {np.min(avgpool_torch):.6f}")
            print(f"Max: {np.max(avgpool_torch):.6f}")
            
            # TF forward pass
            print("\nTensorFlow activations:")
            output_tf = tf_model(dummy_input_tf)
            
            # Compare final outputs
            output_torch = output_torch.numpy()
            output_tf = output_tf.numpy()
            
            max_diff = np.max(np.abs(output_torch - output_tf))
            print(f"\nMaximum difference between PyTorch and TF outputs: {max_diff:.8f}")
            import pdb; pdb.set_trace()
            assert max_diff < 1e-5, f"Outputs differ by more than tolerance (1e-5): {max_diff}"
            print("✓ Outputs match within tolerance!")
