import torch
import tensorflow as tf

from huggingface_hub import hf_hub_download
from fastglioma.inference.run_inference import FastGliomaInferenceSystem

def main():
    
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

if __name__ == "__main__":
    main()
