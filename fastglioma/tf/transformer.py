import tensorflow as tf
from tensorflow import keras
import numpy as np

def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf

class FFPEG(keras.layers.Layer):
    def __init__(self, 
                 embed_dim,
                 dim_ff=96,
                 dim_mlp=36,
                 gamma=0.25,
                 prefix_len=0,
                 pos_emb_grad=True,
                 **kwargs):
        super().__init__()
        self.dim_ff_ = dim_ff
        self.dim_mlp_ = dim_mlp
        self.gamma_ = gamma
        self.embed_dim_ = embed_dim
        self.prefix_len = prefix_len
        
        # Equivalent to PyTorch's nn.Parameter
        self.cls_pos_emb = self.add_weight(
            shape=(1, prefix_len, embed_dim),
            initializer='zeros',
            trainable=True,
            name='cls_pos_emb'
        )
        
        self.num_pos_ = 1
        self.pos_dim_ = 2
        
        self._ff_embed = keras.layers.Dense(
            dim_ff // 2,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=gamma)
        )
        
        if not pos_emb_grad:
            self._ff_embed.trainable = False
            
        self._mlp = keras.Sequential([
            keras.layers.LayerNormalization(epsilon=1e-5),
            keras.layers.Dense(dim_mlp),
            keras.layers.Lambda(gelu),  # Custom GELU implementation
            keras.layers.LayerNormalization(epsilon=1e-5),
            keras.layers.Dense(embed_dim // self.num_pos_)
        ])

    def call(self, H, coords, return_ff=False):
        bsz = tf.shape(H)[0]
        n = tf.shape(H)[1] - self.prefix_len
        
        x = tf.expand_dims(tf.expand_dims(tf.cast(coords, tf.float32), 0), -2)
        
        ff_vec = self._ff_embed(x)
        
        f = tf.concat([tf.cos(ff_vec), tf.sin(ff_vec)], axis=-1)
        f = f / tf.sqrt(float(self.dim_ff_))
        
        if return_ff:
            return f
            
        pe = self._mlp(f)
        pe = tf.reshape(pe, [n, self.embed_dim_])
        pe = tf.expand_dims(pe, 0)
        pe = tf.repeat(pe, bsz, axis=0)
        pe = tf.concat([self.cls_pos_emb, pe], axis=1)
        
        return H + pe


class Attention(keras.layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Initialize weights with same distribution as PyTorch
        self.qkv = keras.layers.Dense(
            dim * 3, 
            use_bias=qkv_bias,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )
        self.attn_drop = keras.layers.Dropout(attn_drop)
        self.proj = keras.layers.Dense(
            dim,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )
        self.proj_drop = keras.layers.Dropout(proj_drop)
        
    def call(self, x, training=False):
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        C = self.dim
        
        # Match PyTorch's exact reshape and permute operations
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # (3, B, num_heads, N, C//num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Exact PyTorch attention computation order
        attn = tf.matmul(q, k, transpose_b=True)  # (B, num_heads, N, N)
        attn = attn * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        
        x = tf.matmul(attn, v)  # (B, num_heads, N, C//num_heads)
        x = tf.transpose(x, [0, 2, 1, 3])  # (B, N, num_heads, C//num_heads)
        x = tf.reshape(x, [B, N, C])
        
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        
        return x, attn


class Block(keras.layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., **kwargs):
        super().__init__()
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)  # Match PyTorch epsilon
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)  # Match PyTorch epsilon
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # Use custom GELU that matches PyTorch exactly
        self.mlp = keras.Sequential([
            keras.layers.Dense(
                mlp_hidden_dim,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'
            ),
            keras.layers.Lambda(gelu),  # Custom GELU implementation
            keras.layers.Dropout(drop),
            keras.layers.Dense(
                dim,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'
            ),
            keras.layers.Dropout(drop)
        ])

    def call(self, x, training=False, return_attention=False):
        norm_x = self.norm1(x)
        y, attn = self.attn(norm_x, training=training)
        if return_attention:
            return attn
            
        x = x + y
        residual = x
        
        x = self.mlp(self.norm2(x), training=training)
        x = residual + x
        
        return x


class TransformerMIL(keras.Model):
    def __init__(self, 
                 global_pool='token',
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 pos_emb_type=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 **kwargs):
        super().__init__()
        self.global_pool = global_pool
        assert self.global_pool in ['', 'avg', 'token']
        
        self.cls_token = self.add_weight(
            shape=(1, kwargs.get("prefix_len", 1), embed_dim),
            initializer='zeros',
            trainable=True,
            name='cls_token'
        )
        
        self.pos_embed = keras.layers.Lambda(lambda x: x)
        if pos_emb_type:
            self.pos_embed = FFPEG(
                embed_dim=embed_dim,
                **kwargs
            )

        self.blocks = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            ) for _ in range(depth)
        ]
        
        self.norm = keras.layers.LayerNormalization(epsilon=1e-5)
        self.dim_out = embed_dim

    def call(self, x, coords=None, training=False):
        if len(tf.shape(x)) == 2:
            x = tf.expand_dims(x, 0)
            
        batch_size = tf.shape(x)[0]
        x = tf.concat([tf.tile(self.cls_token, [batch_size, 1, 1]), x], axis=1)
        
        if coords is not None:
            x = self.pos_embed(x, coords)
        else:
            x = self.pos_embed(x)
            
        for block in self.blocks:
            x = block(x, training=training)
            
        x = self.norm(x)

        if self.global_pool:
            x = tf.reduce_mean(x[:, 1:], axis=1) if self.global_pool == 'avg' else x[:, 0]
            
        return x


def convert_pytorch_model_to_tf(pytorch_model, tf_model):
    # Build model first
    batch_size = 1
    n_patches = 16
    dummy_input = tf.zeros((batch_size, n_patches, pytorch_model.dim_out), dtype=tf.float32)
    dummy_coords = tf.zeros((1, n_patches, 2), dtype=tf.float32)
    _ = tf_model(dummy_input, coords=dummy_coords)
    
    state_dict = pytorch_model.state_dict()
    tf_weights = []
    
    # 1. cls_pos_emb (1, 8, 512)
    tf_weights.append(state_dict['pos_embed.cls_pos_emb'].numpy())
    
    # 2. ffpeg/dense/kernel (2, 48)
    tf_weights.append(state_dict['pos_embed._ff_embed.weight'].numpy().transpose())
    
    # 3-10. FFPEG MLP weights
    tf_weights.append(state_dict['pos_embed._mlp.0.weight'].numpy())  # layer_norm gamma (96,)
    tf_weights.append(state_dict['pos_embed._mlp.0.bias'].numpy())   # layer_norm beta (96,)
    tf_weights.append(state_dict['pos_embed._mlp.1.weight'].numpy().transpose())  # dense_1 kernel (96, 36)
    tf_weights.append(state_dict['pos_embed._mlp.1.bias'].numpy())   # dense_1 bias (36,)
    tf_weights.append(state_dict['pos_embed._mlp.3.weight'].numpy())  # layer_norm_1 gamma (36,)
    tf_weights.append(state_dict['pos_embed._mlp.3.bias'].numpy())   # layer_norm_1 beta (36,)
    tf_weights.append(state_dict['pos_embed._mlp.4.weight'].numpy().transpose())  # dense_2 kernel (36, 512)
    tf_weights.append(state_dict['pos_embed._mlp.4.bias'].numpy())   # dense_2 bias (512,)
    
    # Map transformer blocks
    for i in range(len(pytorch_model.blocks)):
        # Layer norm 2/4 (512,)
        tf_weights.append(state_dict[f'blocks.{i}.norm1.weight'].numpy())
        tf_weights.append(state_dict[f'blocks.{i}.norm1.bias'].numpy())
        
        # Attention weights
        qkv_weight = state_dict[f'blocks.{i}.attn.qkv.weight'].numpy()
        tf_weights.append(qkv_weight.transpose())  # dense_3/7 kernel (512, 1536)
        
        proj_weight = state_dict[f'blocks.{i}.attn.proj.weight'].numpy()
        tf_weights.append(proj_weight.transpose())  # dense_4/8 kernel (512, 512)
        tf_weights.append(state_dict[f'blocks.{i}.attn.proj.bias'].numpy())  # dense_4/8 bias (512,)
        
        # Layer norm 3/5 (512,)
        tf_weights.append(state_dict[f'blocks.{i}.norm2.weight'].numpy())
        tf_weights.append(state_dict[f'blocks.{i}.norm2.bias'].numpy())
        
        # MLP weights
        tf_weights.append(state_dict[f'blocks.{i}.mlp.fc1.weight'].numpy().transpose())  # dense_5/9 kernel (512, 2048)
        tf_weights.append(state_dict[f'blocks.{i}.mlp.fc1.bias'].numpy())  # dense_5/9 bias (2048,)
        tf_weights.append(state_dict[f'blocks.{i}.mlp.fc2.weight'].numpy().transpose())  # dense_6/10 kernel (2048, 512)
        tf_weights.append(state_dict[f'blocks.{i}.mlp.fc2.bias'].numpy())  # dense_6/10 bias (512,)
    
    # Final layer norm (512,)
    tf_weights.append(state_dict['norm.weight'].numpy())
    tf_weights.append(state_dict['norm.bias'].numpy())
    
    # cls_token (1, 8, 512)
    tf_weights.append(state_dict['cls_token'].numpy())
    
    # Verify shapes match
    print("\nVerifying weight shapes:")
    for i, (w, tw) in enumerate(zip(tf_model.weights, tf_weights)):
        print(f"{i}: {w.name} - Expected: {w.shape}, Got: {tw.shape}")
        assert w.shape == tw.shape, f"Shape mismatch for {w.name}: Expected {w.shape}, got {tw.shape}"
    
    tf_model.set_weights(tf_weights)
    return tf_model


if __name__ == "__main__":
    import torch
    from torch import nn
    from huggingface_hub import hf_hub_download
    from fastglioma.inference.run_inference import FastGliomaInferenceSystem

    # Load models
    cf = {
        "infra": {
            "log_dir": "/nfs/turbo/umms-tocho-snr/exp/akhilk/",
            "exp_name": "fastglioma_dev/",
            "comment": "inference",
            "hf_repo": "mlinslab/fastglioma",
            "seed": 1000
        },
        "data": {
            "db_root": "/nfs/turbo/umms-tocho/data/opensrh/",
            "studies": ["NIO_001"],
            "patch_input": "highres",
            "use_patient_class": True
        },
        "model": {
            "patch": {
                "backbone": {
                    "which": "resnet34",
                    "params": {
                        "num_channel_in": 3
                    }
                },
                "mlp_hidden": [],
                "num_embedding_out": 128
            },
            "slide": {
                "mil": {
                    "which": "transformer",
                    "params": {
                        "embed_dim": 512,
                        "depth": 2,
                        "num_heads": 4,
                        "pos_emb_type": "FFPEG",
                        "pos_emb_grad": True,
                        "prefix_len": 8
                    }
                },
                "mlp_hidden": [512]
            }
        },
        "eval": {
            "predict_batch_size": 4,
            "ckpt_path": "fastglioma_highres_model.ckpt"
        }
    }

    # Load model from huggingface repo
    ckpt_path = hf_hub_download(repo_id=cf["infra"]["hf_repo"],
                                filename=cf["eval"]["ckpt_path"])
    pl_system = FastGliomaInferenceSystem.load_from_checkpoint(ckpt_path,
                                                            cf=cf,
                                                            num_it_per_ep=0)
    pytorch_model = pl_system.model.bb.mil

    # Create TF model with matching architecture
    tf_model = TransformerMIL(
        embed_dim=512,
        depth=2,
        num_heads=4,
        pos_emb_type="FFPEG",
        pos_emb_grad=True,
        prefix_len=8
    )

    # Convert model
    tf_model = convert_pytorch_model_to_tf(pytorch_model, tf_model)

    # Create dummy data for verification
    batch_size = 1
    n_patches = 16
    embed_dim = pytorch_model.dim_out
    
    # Create identical numpy arrays for both frameworks
    dummy_input_np = np.random.randn(batch_size, n_patches, embed_dim).astype(np.float32)
    dummy_coords_np = np.random.randn(1, n_patches, 2).astype(np.float32)
    
    # Convert to framework-specific tensors
    dummy_input_torch = torch.from_numpy(dummy_input_np)
    dummy_coords_torch = torch.from_numpy(dummy_coords_np)
    
    dummy_input_tf = tf.convert_to_tensor(dummy_input_np, dtype=tf.float32)
    dummy_coords_tf = tf.convert_to_tensor(dummy_coords_np, dtype=tf.float32)
    
    # Get outputs from both models
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input_torch, coords=dummy_coords_torch).numpy()
    
    tf_output = tf_model(dummy_input_tf, coords=dummy_coords_tf).numpy()

    # Compare outputs
    print("\nComparing outputs:")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"TensorFlow output shape: {tf_output.shape}")
    print(f"Max absolute difference: {np.max(np.abs(pytorch_output - tf_output))}")
    
    # Assert outputs are close
    np.testing.assert_allclose(pytorch_output, tf_output, rtol=1e-5, atol=1e-5)
    print("✓ Outputs match within tolerance!")

    import pdb; pdb.set_trace()

