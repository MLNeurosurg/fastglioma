"""AdditiveMIL in TensorFlow.

Matches PyTorch implementation exactly.

Copyright (c) 2024 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import tensorflow as tf
from tensorflow import keras
import logging


class TruncatedNormalInitializer(keras.initializers.Initializer):
    """Truncated normal initialization matching PyTorch's trunc_normal_"""
    
    def __init__(self, mean=0.0, std=1.0, a=-2.0, b=2.0):
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b
    
    def __call__(self, shape, dtype=None):
        # Generate values from a normal distribution and clip to match PyTorch's trunc_normal_
        # PyTorch's trunc_normal_ samples from a normal distribution and clips to [a, b]
        values = tf.random.normal(shape, mean=self.mean, stddev=self.std, dtype=dtype)
        # Clip to [mean + a*std, mean + b*std] range
        values = tf.clip_by_value(values, self.mean + self.a * self.std, self.mean + self.b * self.std)
        return values


class MLP(keras.layers.Layer):
    """MLP layer matching PyTorch implementation."""
    
    def __init__(self, n_in, n_out, hidden_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.n_in = n_in
        self.n_out = n_out
        self.hidden_layers = hidden_layers or []
        
        # Build layers: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear (no ReLU at end)
        layers_in = [n_in] + self.hidden_layers
        layers_out = self.hidden_layers + [n_out]
        
        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(layers_in, layers_out)):
            self.layers.append(
                keras.layers.Dense(
                    out_dim,
                    use_bias=True,
                    kernel_initializer=keras.initializers.GlorotUniform(),
                    bias_initializer=keras.initializers.Constant(0.01),
                    name=f'dense_{i}'
                )
            )
            # Add ReLU except for the last layer
            if i < len(layers_in) - 1:
                self.layers.append(keras.layers.ReLU(name=f'relu_{i}'))
    
    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
        return x


class TFAdditiveMIL(keras.Model):
    """Additive MIL layer matching PyTorch implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Fixed parameters based on config
        self.dim = 512
        self.hidden_dim = 512
        self.W_out = 1
        self.mlp_hidden_dims = [512]
        self.mlp_out_dim = 1
        self.drop_rate = 0.2
        self.init_std = 0.02
        self.use_gating = True
        self.use_norm = True
        
        # Truncated normal initializer
        trunc_init = TruncatedNormalInitializer(std=self.init_std)
        
        # Additive attention network (V)
        self.attention_V = keras.layers.Dense(
            self.hidden_dim,
            use_bias=True,
            kernel_initializer=trunc_init,
            bias_initializer='zeros',
            name='attention_V'
        )
        
        # Gating network (V)
        if self.use_gating:
            self.gating_V = keras.layers.Dense(
                self.hidden_dim,
                use_bias=True,
                kernel_initializer=trunc_init,
                bias_initializer='zeros',
                name='gating_V'
            )
        else:
            self.gating_V = None
        
        # Final projection to W_out attention channels
        self.W = keras.layers.Dense(
            self.W_out,
            use_bias=True,
            kernel_initializer=trunc_init,
            bias_initializer='zeros',
            name='W'
        )
        
        self.dropout = keras.layers.Dropout(self.drop_rate)
        
        # Layer normalization
        if self.use_norm:
            self.norm_attn = keras.layers.LayerNormalization(epsilon=1e-6, name='norm_attn')
            self.norm_mlp = keras.layers.LayerNormalization(epsilon=1e-6, name='norm_mlp')
        else:
            self.norm_attn = None
            self.norm_mlp = None
        
        # Per-patch MLP that produces logits/features to pool
        self.mlp = MLP(
            n_in=self.dim,
            n_out=self.mlp_out_dim,
            hidden_layers=self.mlp_hidden_dims,
            name='mlp'
        )
    
    def call(self, H, training=False, return_attention=False, return_features=False):
        """
        H: [N, dim] for a *single* bag (or [batch_size, N, dim] for batched).
        Returns:
            bag_logits: [1, W_out] or [batch_size, W_out]
            A:          [W_out, N] or [batch_size, W_out, N] (attention maps per head)
        """
        # Handle batched vs single bag
        batched = len(tf.shape(H)) == 3
        if batched:
            batch_size = tf.shape(H)[0]
            N = tf.shape(H)[1]
        else:
            # Single bag: add batch dimension
            H = tf.expand_dims(H, 0)
            batch_size = 1
            N = tf.shape(H)[1]
        
        # Pre-norm for attention path
        if self.norm_attn is not None:
            H_attn = self.norm_attn(H, training=training)
        else:
            H_attn = H
        
        # Pre-norm for MLP path
        if self.norm_mlp is not None:
            H_mlp = self.norm_mlp(H, training=training)
        else:
            H_mlp = H
        
        # Dropout after norm
        H_attn = self.dropout(H_attn, training=training)
        H_mlp = self.dropout(H_mlp, training=training)
        
        # Additive + gated attention
        attn_feat = tf.tanh(self.attention_V(H_attn))  # [batch_size, N, hidden_dim]
        if self.gating_V is not None:
            gate_feat = tf.sigmoid(self.gating_V(H_attn))  # [batch_size, N, hidden_dim]
            score_feat = attn_feat * gate_feat
        else:
            score_feat = attn_feat
        
        scores = self.W(score_feat)  # [batch_size, N, W_out]
        
        # Per-patch logits/features to be pooled
        patch_logits = self.mlp(H_mlp, training=training)  # [batch_size, N, W_out]
        
        # Compute attention over *instances* for each head/class:
        # scores: [batch_size, N, W_out] -> [batch_size, W_out, N]
        scores_T = tf.transpose(scores, [0, 2, 1])  # [batch_size, W_out, N]
        
        A = tf.nn.softmax(scores_T, axis=2)  # [batch_size, W_out, N]
        
        # Pool per-patch logits with per-head attention
        # A: [batch_size, W_out, N], patch_logits^T: [batch_size, W_out, N]
        patch_logits_T = tf.transpose(patch_logits, [0, 2, 1])  # [batch_size, W_out, N]
        bag_logits = tf.reduce_sum(A * patch_logits_T, axis=2)  # [batch_size, W_out]
        
        # Match PyTorch behavior: always keep batch dimension
        # For single bag, shape is [1, W_out] to match PyTorch's unsqueeze(0)
        # For batched, shape is [batch_size, W_out]
        
        if return_features:
            if not batched:
                # Remove batch dimension for single bag case
                A = tf.squeeze(A, axis=0)  # [W_out, N]
                patch_logits_T = tf.squeeze(patch_logits_T, axis=0)  # [W_out, N]
            return bag_logits, A, patch_logits_T
        elif return_attention:
            if not batched:
                A = tf.squeeze(A, axis=0)  # [W_out, N]
            return bag_logits, A
        else:
            return bag_logits
    
    def get_attention_maps(self, H, training=False):
        """Get attention maps only."""
        _, A = self.call(H, training=training, return_attention=True)
        return A


def convert_pytorch_addmil_to_tf(pytorch_model, tf_model):
    """Convert PyTorch AdditiveMIL weights to TensorFlow model.
    
    Weight order in TensorFlow model:
    1. attention_V: kernel (transposed), bias
    2. gating_V: kernel (transposed), bias
    3. W: kernel (transposed), bias
    4. norm_attn: gamma (weight), beta (bias)
    5. norm_mlp: gamma (weight), beta (bias)
    6. mlp: dense layers (kernels transposed, biases)
    """
    # Build model first
    batch_size = 1
    n_patches = 16
    dummy_input = tf.zeros((batch_size, n_patches, pytorch_model.dim), dtype=tf.float32)
    _ = tf_model(dummy_input)

    state_dict = pytorch_model.state_dict()
    tf_weights = []

    # 1. attention_V/kernel (512, 512) - PyTorch: (hidden_dim, dim), TF: (dim, hidden_dim)
    tf_weights.append(state_dict['attention_V.weight'].numpy().transpose())
    
    # 2. attention_V/bias (512,)
    tf_weights.append(state_dict['attention_V.bias'].numpy())
    
    # 3. gating_V/kernel (512, 512) - PyTorch: (hidden_dim, dim), TF: (dim, hidden_dim)
    if pytorch_model.use_gating:
        tf_weights.append(state_dict['gating_V.weight'].numpy().transpose())
        
        # 4. gating_V/bias (512,)
        tf_weights.append(state_dict['gating_V.bias'].numpy())
    
    # 5. W/kernel (1, 512) - PyTorch: (W_out, hidden_dim), TF: (hidden_dim, W_out)
    tf_weights.append(state_dict['W.weight'].numpy().transpose())
    
    # 6. W/bias (1,)
    tf_weights.append(state_dict['W.bias'].numpy())
    
    # 7-8. norm_attn: gamma (weight) and beta (bias) - both (512,)
    if pytorch_model.use_norm:
        tf_weights.append(state_dict['norm_attn.weight'].numpy())
        tf_weights.append(state_dict['norm_attn.bias'].numpy())
        
        # 9-10. norm_mlp: gamma (weight) and beta (bias) - both (512,)
        tf_weights.append(state_dict['norm_mlp.weight'].numpy())
        tf_weights.append(state_dict['norm_mlp.bias'].numpy())
    
    # 11-14. MLP layers
    # MLP structure: Linear(512, 512) -> ReLU -> Linear(512, 1)
    # So we have: layers.0 (Dense), layers.1 (ReLU - no weights), layers.2 (Dense)
    
    # 11. mlp dense_0/kernel (512, 512) - PyTorch: (out_dim, in_dim), TF: (in_dim, out_dim)
    tf_weights.append(state_dict['mlp.layers.0.weight'].numpy().transpose())
    
    # 12. mlp dense_0/bias (512,)
    tf_weights.append(state_dict['mlp.layers.0.bias'].numpy())
    
    # 13. mlp dense_1/kernel (1, 512) - PyTorch: (out_dim, in_dim), TF: (in_dim, out_dim)
    # Note: layer 1 is ReLU, so layer 2 is the second Dense
    tf_weights.append(state_dict['mlp.layers.2.weight'].numpy().transpose())
    
    # 14. mlp dense_1/bias (1,)
    tf_weights.append(state_dict['mlp.layers.2.bias'].numpy())

    # Verify shapes match
    logging.debug("\nVerifying weight shapes:")
    for i, (w, tw) in enumerate(zip(tf_model.weights, tf_weights)):
        logging.debug(f"{i}: {w.name} - Expected: {w.shape}, Got: {tw.shape}")
        assert w.shape == tw.shape, f"Shape mismatch for {w.name}: Expected {w.shape}, got {tw.shape}"

    tf_model.set_weights(tf_weights)
    return tf_model