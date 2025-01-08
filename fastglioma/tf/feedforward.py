import os
import yaml
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
import logging

import torch
import tensorflow as tf
from tensorflow import keras
import numpy as np

from huggingface_hub import hf_hub_download
from fastglioma.inference.run_inference import FastGliomaInferenceSystem
from fastglioma.datasets.srh_dataset import SlideDataset, slide_collate_fn
from fastglioma.datasets.improc import get_srh_base_aug, get_strong_aug
from torchvision.transforms import Compose

from fastglioma.utils.common import (parse_args, get_exp_name, config_loggers,
                                     get_num_worker)
from fastglioma.inference.run_inference import setup_eval_paths


def create_tf_dataset(cf: Dict[str, Any]):
    """Create TensorFlow dataset from PyTorch dataset"""

    def get_transform(cf):
        return Compose(
            get_srh_base_aug(
                base_aug=("three_channels" if cf["data"]["patch_input"] ==
                          "highres" else "ch2_only")) +
            (get_strong_aug([{
                "which": "inpaint_rows_always_apply",
                "params": {
                    "image_size": 300,
                    "y_skip": 5
                }
            }], 1.) if cf["data"]["patch_input"] == "lowres" else []))

    # Use the existing PyTorch dataset
    dset = SlideDataset(data_root=cf["data"]["db_root"],
                        studies=cf["data"]["studies"],
                        transform=get_transform(cf),
                        balance_slide_per_class=False,
                        use_patient_class=cf["data"]["use_patient_class"])

    # Create PyTorch dataloader
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        collate_fn=slide_collate_fn,
        num_workers=get_num_worker())

    return loader, dset.class_to_idx_


def process_predictions(predictions: Dict, class_to_idx: Dict) -> Dict:
    """Process predictions similar to PyTorch version"""
    pred = {}

    # Convert logits to probabilities
    pred["logits"] = predictions["logits"]
    pred["scores"] = tf.sigmoid(pred["logits"]).numpy()

    # Convert numeric labels back to strings
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pred["label"] = [idx_to_class[l] for l in predictions["labels"]]

    # Get slide names
    pred["slide"] = ["/".join(imp[0][0].split("/")[:9]) for imp in predictions["paths"]] #yapf:disable

    # Store embeddings
    pred["embeddings"] = predictions["embeddings"]

    # Sort predictions by slide name
    sorted_indices = sorted(range(len(pred['slide'])),
                            key=lambda k: pred['slide'][k])

    # Apply sorting to all fields
    for key in pred:
        if isinstance(pred[key], list):
            pred[key] = [pred[key][i] for i in sorted_indices]
        elif isinstance(pred[key], np.ndarray):
            pred[key] = pred[key][sorted_indices]

    return pred


def get_tf_predictions(cf: Dict[str, Any],
                       model_dict: Dict[str, tf.keras.Model]) -> Dict:
    """Run inference using TensorFlow models"""

    loader, class_to_idx = create_tf_dataset(cf)

    all_predictions = {
        "logits": [],
        "embeddings": [],
        "labels": [],
        "paths": []
    }

    # Run inference
    for batch in tqdm(loader):
        # Get first view of images and coordinates
        images = tf.cast(np.transpose(batch["image"][0][0].numpy(),
                                      (0, 2, 3, 1)),
                         dtype=tf.float32)
        coords = tf.cast(tf.expand_dims(batch["coords"][0][0].numpy(), axis=0),
                         dtype=tf.float32)

        # Forward pass through models
        patch_embeddings = tf.squeeze(model_dict["resnet"](images))
        patch_embeddings = tf.expand_dims(patch_embeddings, axis=0)

        slide_embedding = model_dict["transformer"](patch_embeddings,
                                                    coords=coords)
        logits = model_dict["head"](slide_embedding)

        # Store predictions
        all_predictions["logits"].append(logits.numpy())
        all_predictions["embeddings"].append(slide_embedding.numpy())
        all_predictions["labels"].append(batch["label"].numpy())
        all_predictions["paths"].append(batch["path"])

    # Concatenate results
    for key in ["logits", "embeddings", "labels"]:
        all_predictions[key] = np.concatenate(all_predictions[key])

    # Process predictions into final format
    predictions = process_predictions(all_predictions, class_to_idx)
    return predictions


def compare_pytorch_tensorflow_outputs(pl_system,
                                       model_dict,
                                       num_channels=3,
                                       batch_size=4,
                                       num_patches=10):
    """Compare outputs between PyTorch and TensorFlow models using dummy data.
    
    Args:
        pl_system: PyTorch Lightning system containing the PyTorch models
        model_dict: Dictionary containing TensorFlow models
        batch_size: Number of samples in batch
        num_patches: Number of patches per sample
    """
    # Create dummy data
    dummy_images = torch.randn(batch_size, num_patches, num_channels, 224, 224)
    dummy_coords = torch.randn(batch_size, num_patches, 2)

    # PyTorch forward pass
    with torch.no_grad():
        # Convert to expected format and run through models
        pt_patch_embeddings = pl_system.model.backbone(dummy_images.view(-1, num_channels, 224, 224)) #yapf:disable
        pt_patch_embeddings = pt_patch_embeddings.view(batch_size, num_patches,
                                                       -1)
        pt_slide_embedding = pl_system.model.bb.mil(pt_patch_embeddings,
                                                    coords=dummy_coords)
        pt_logits = pl_system.model.head(pt_slide_embedding)

    # TensorFlow forward pass
    # Reshape and transpose images for TF format (B*N, H, W, C)
    tf_images = tf.transpose(dummy_images.numpy(), (0, 1, 3, 4, 2))
    tf_images = tf.reshape(tf_images, (-1, 224, 224, num_channels))
    tf_coords = tf.cast(dummy_coords.numpy(), dtype=tf.float32)

    # Run through TF models
    tf_patch_embeddings = tf.squeeze(model_dict["resnet"](tf_images))
    tf_patch_embeddings = tf.reshape(tf_patch_embeddings,
                                     (batch_size, num_patches, -1))
    tf_slide_embedding = model_dict["transformer"](tf_patch_embeddings,
                                                   coords=tf_coords)
    tf_logits = model_dict["head"](tf_slide_embedding)

    # Compare outputs
    print("\nOutput Comparisons:")
    print("------------------")
    print(f"Patch Embeddings - Max Diff: {np.max(np.abs(pt_patch_embeddings.numpy() - tf_patch_embeddings.numpy())):.6f}") #yapf:disable
    print(f"Slide Embeddings - Max Diff: {np.max(np.abs(pt_slide_embedding.numpy() - tf_slide_embedding.numpy())):.6f}") #yapf:disable
    print(f"Logits - Max Diff: {np.max(np.abs(pt_logits.numpy() - tf_logits.numpy())):.6f}") #yapf:disable


def main():
    """Driver script for inference pipeline."""
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, pred_dir, cp_config = setup_eval_paths(cf, get_exp_name, "")

    torch.manual_seed(cf["infra"]["seed"])
    tf.random.set_seed(cf["infra"]["seed"])

    # Logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    # Load PyTorch model and convert to TF
    ckpt_path = hf_hub_download(repo_id=cf["infra"]["hf_repo"],
                                filename=cf["eval"]["ckpt_path"])
    pl_system = FastGliomaInferenceSystem.load_from_checkpoint(ckpt_path,
                                                               cf=cf,
                                                               num_it_per_ep=0)

    # Convert models to TensorFlow
    from fastglioma.tf.resnet import resnet_backbone, convert_resnet_weights
    from fastglioma.tf.transformer import TransformerMIL, convert_pytorch_transformer_to_tf

    # Initialize TF models
    tf_resnet = resnet_backbone(arch=cf["model"]["patch"]["backbone"]["which"],
                                **cf["model"]["patch"]["backbone"]["params"])
    tf_transformer = TransformerMIL(**cf["model"]["slide"]["mil"]["params"])

    tf_head = []

    if len(cf["model"]["slide"].get("mlp_hidden", [])) > 0:
        for hidden_dim in cf["model"]["slide"]["mlp_hidden"]:
            tf_head.append(keras.layers.Dense(hidden_dim, use_bias=True))
            tf_head.append(keras.layers.ReLU())
        tf_head.append(keras.layers.Dense(1, use_bias=True))

    tf_head = keras.Sequential(tf_head)

    # Convert weights
    torch_resnet = pl_system.model.backbone.eval()
    torch_transformer = pl_system.model.bb.mil.eval()
    torch_head = pl_system.model.head.eval()

    tf_resnet = convert_resnet_weights(
        torch_resnet,
        tf_resnet,
        num_channel_in=cf["model"]["patch"]["backbone"]["params"].get(
            "num_channel_in", 3))
    tf_transformer = convert_pytorch_transformer_to_tf(torch_transformer,
                                                       tf_transformer)

    # Convert head weights
    dummy_input = tf.zeros((1, 512), dtype=tf.float32)
    _ = tf_head(dummy_input)
    state_dict = torch_head.state_dict()
    tf_weights = [
        state_dict['layers.0.weight'].numpy().transpose(),
        state_dict['layers.0.bias'].numpy(),
        state_dict['layers.2.weight'].numpy().transpose(),
        state_dict['layers.2.bias'].numpy()
    ]
    tf_head.set_weights(tf_weights)

    # Create model dictionary
    model_dict = {
        "resnet": tf_resnet,
        "transformer": tf_transformer,
        "head": tf_head
    }

    if cf["eval"]["compare_to_torch"]:
        comparison_results = compare_pytorch_tensorflow_outputs(
            pl_system,
            model_dict,
            num_channels=cf["model"]["patch"]["backbone"]["params"].get(
                "num_channel_in", 3))

    # Run inference
    predictions = get_tf_predictions(cf, model_dict)
    torch.save(predictions, os.path.join(pred_dir, "tf_predictions.pt"))


if __name__ == "__main__":
    main()
