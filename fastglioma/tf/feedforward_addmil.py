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
        # Get first view of images
        images = tf.cast(np.transpose(batch["image"][0][0].numpy(),
                                      (0, 2, 3, 1)),
                         dtype=tf.float32)

        # Forward pass through models
        patch_embeddings = tf.squeeze(model_dict["resnet"](images))
        # AdditiveMIL expects [N, dim] for single bag or [batch_size, N, dim] for batched
        # We have [N, dim] from a single slide, so add batch dimension
        patch_embeddings = tf.expand_dims(patch_embeddings, axis=0)  # [1, N, dim]

        # AdditiveMIL doesn't use coordinates
        # AdditiveMIL outputs [1, W_out] = [1, 1] for single bag (bag logits directly)
        logits = model_dict["addmil"](patch_embeddings)  # [1, W_out] = [1, 1]
        slide_embedding = logits  # Store as embedding for consistency (same as logits for AdditiveMIL)

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


def compare_pytorch_tensorflow_outputs(pt_models,
                                       tf_models,
                                       num_channels=3,
                                       batch_size=4,
                                       num_patches=10):
    """Compare outputs between PyTorch and TensorFlow models using dummy data.
    
    Args:
        pt_models: Dictionary containing PyTorch models (backbone, mil)
        tf_models: Dictionary containing TensorFlow models (resnet, addmil)
        batch_size: Number of samples in batch
        num_patches: Number of patches per sample
    """
    # Create dummy data
    dummy_images = torch.randn(batch_size, num_patches, num_channels, 224, 224)

    # PyTorch forward pass
    with torch.no_grad():
        # Convert to expected format and run through models
        pt_patch_embeddings = pt_models["backbone"](dummy_images.view(-1, num_channels, 224, 224)) #yapf:disable
        pt_patch_embeddings = pt_patch_embeddings.view(batch_size, num_patches,
                                                       -1)
        # AdditiveMIL doesn't use coordinates - process each bag separately
        pt_logits_list = []
        for i in range(batch_size):
            bag_logits = pt_models["mil"](pt_patch_embeddings[i])
            pt_logits_list.append(bag_logits.squeeze(0))
        pt_logits = torch.stack(pt_logits_list)  # [batch_size, W_out]
        pt_slide_embedding = pt_logits  # For AdditiveMIL, embeddings = logits

    # TensorFlow forward pass
    # Reshape and transpose images for TF format (B*N, H, W, C)
    tf_images = tf.transpose(dummy_images.numpy(), (0, 1, 3, 4, 2))
    tf_images = tf.reshape(tf_images, (-1, 224, 224, num_channels))

    # Run through TF models
    tf_patch_embeddings = tf.squeeze(tf_models["resnet"](tf_images))
    tf_patch_embeddings = tf.reshape(tf_patch_embeddings,
                                     (batch_size, num_patches, -1))
    # AdditiveMIL doesn't use coordinates and outputs logits directly
    tf_logits = tf_models["addmil"](tf_patch_embeddings)  # [batch_size, W_out]
    tf_slide_embedding = tf_logits  # For AdditiveMIL, embeddings = logits

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

    bb_ckpt = "/nfs/turbo/umms-tocho-snr/exp/akhilk/fast_glioma/patch_encoder/patch_encoder/a955fdfd_Nov17-09-42-47_sd1000_hidisc_skip_3ch_tune0/models/ckpt-epoch799-step18400-loss0.00.ckpt"
    ckpt_path = "/nfs/turbo/umms-tocho-snr/exp/akhilk/fastglioma_v2/120425_split/ordmet/68545fc3_Dec04-14-26-41_ordmet_a955fdfd_addmil.512_lr3e-4_wd5e-2_drop0.2_bsz64_ep100_noupsm_fullsrh_fast_glioma_tune0/models/ckpt-epoch24-step1025-loss0.00.ckpt"
    
    logger.info(f"Loading backbone checkpoint from: {bb_ckpt}")
    logger.info(f"Loading AdditiveMIL checkpoint from: {ckpt_path}")
    
    # Load checkpoint state dicts
    bb_ckpt_dict = torch.load(bb_ckpt, map_location="cpu")
    mil_ckpt_dict = torch.load(ckpt_path, map_location="cpu")
    
    bb_state_dict = bb_ckpt_dict["state_dict"]
    mil_state_dict = mil_ckpt_dict["state_dict"]
    
    # Manually instantiate PyTorch models
    from fastglioma.models.resnet import resnet_backbone
    from fastglioma.models.mil import AdditiveMIL
    
    # Initialize PyTorch backbone
    pt_backbone = resnet_backbone(
        arch=cf["model"]["patch"]["backbone"]["which"],
        num_channel_in=cf["model"]["patch"]["backbone"]["params"].get("num_channel_in", 3)
    )
    
    # Initialize PyTorch AdditiveMIL
    pt_addmil = AdditiveMIL(**cf["model"]["slide"]["mil"]["params"])
    
    # Load backbone weights from bb_ckpt
    # Extract backbone weights with prefix "model.bb."
    backbone_state_dict = {k.removeprefix("model.bb."): v 
                          for k, v in bb_state_dict.items() 
                          if k.startswith("model.bb.")}
    
    # Load AdditiveMIL weights from ckpt_path
    # Extract AdditiveMIL weights with prefix "model.bb.mil."
    addmil_state_dict = {k.removeprefix("model.bb.mil."): v 
                        for k, v in mil_state_dict.items() 
                        if k.startswith("model.bb.mil.")}
    
    pt_backbone.load_state_dict(backbone_state_dict, strict=False)
    pt_addmil.load_state_dict(addmil_state_dict, strict=False)
    
    pt_backbone.eval()
    pt_addmil.eval()
    
    pt_models = {
        "backbone": pt_backbone,
        "mil": pt_addmil
    }

    # Convert models to TensorFlow
    from fastglioma.tf.resnet import resnet_backbone as tf_resnet_backbone, convert_resnet_weights
    from fastglioma.tf.addmil import TFAdditiveMIL, convert_pytorch_addmil_to_tf

    # Initialize TF models
    tf_resnet = tf_resnet_backbone(arch=cf["model"]["patch"]["backbone"]["which"],
                                   **cf["model"]["patch"]["backbone"]["params"])
    tf_addmil = TFAdditiveMIL()

    # Convert weights from PyTorch to TensorFlow
    tf_resnet = convert_resnet_weights(
        pt_backbone,
        tf_resnet,
        num_channel_in=cf["model"]["patch"]["backbone"]["params"].get(
            "num_channel_in", 3))
    tf_addmil = convert_pytorch_addmil_to_tf(pt_addmil, tf_addmil)
    
    tf_models = {
        "resnet": tf_resnet,
        "addmil": tf_addmil
    }

    if cf["eval"]["compare_to_torch"]:
        comparison_results = compare_pytorch_tensorflow_outputs(
            pt_models,
            tf_models,
            num_channels=cf["model"]["patch"]["backbone"]["params"].get(
                "num_channel_in", 3))

    # Save the TensorFlow models
    models_save_dir = os.path.join(exp_root, "tf_models")
    os.makedirs(models_save_dir, exist_ok=True)
    
    # Save weights for all models
    for model_name, model in tf_models.items():
        weights_path = os.path.join(models_save_dir, f"{model_name}_weights")
        model.save_weights(weights_path)
        logger.info(f"Saved {model_name} weights to {weights_path}")
        
    # Run inference
    predictions = get_tf_predictions(cf, tf_models)
    torch.save(predictions, os.path.join(pred_dir, "tf_predictions.pt"))


if __name__ == "__main__":
    main()
