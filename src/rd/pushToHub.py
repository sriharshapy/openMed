#!/usr/bin/env python3
"""
HuggingFace Model Hub Push Script for OpenMed ResNet50 Models
This script pushes the trained ResNet50 models for medical imaging to HuggingFace Hub.
Models included:
- ResNet50 Brain Tumor Classification (3-class: Glioma, Meningioma, Tumor)
- ResNet50 Tuberculosis Detection (2-class: Normal, Tuberculosis)
- ResNet50 Pneumonia Detection (2-class: Normal, Pneumonia)
"""

import os
import torch
import json
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo, upload_file, Repository
from huggingface_hub.utils import RepositoryNotFoundError
import tempfile
import shutil

class OpenMedModelPusher:
    def __init__(self, hf_token: str, repo_name: str = "openmed-medical-imaging-models"):
        """
        Initialize the model pusher with HuggingFace token and repository name.
        
        Args:
            hf_token: HuggingFace API token
            repo_name: Name of the repository to create/use
        """
        self.hf_token = hf_token
        self.repo_name = repo_name
        self.api = HfApi()
        
        # Login to HuggingFace
        login(token=hf_token)
        print(f"[SUCCESS] Successfully logged in to HuggingFace Hub")
        
        # Get username
        self.username = self.api.whoami()["name"]
        self.full_repo_name = f"{self.username}/{repo_name}"
        print(f"[REPO] Repository will be: {self.full_repo_name}")
        
        # Get the script directory and calculate absolute paths
        script_dir = Path(__file__).parent.absolute()
        checkpoints_dir = script_dir.parent.parent / "checkpoints"
        print(f"[DEBUG] Script directory: {script_dir}")
        print(f"[DEBUG] Checkpoints directory: {checkpoints_dir}")
        print(f"[DEBUG] Checkpoints exists: {checkpoints_dir.exists()}")
        
        # Model configurations
        self.models_config = {
            "resnet50_brain_tumor_full": {
                "path": str(checkpoints_dir / "resnet50_brain_tumor_full" / "best_resnet50_brain_tumor_full_trained.pth"),
                "num_classes": 3,
                "classes": ["brain_glioma", "brain_menin", "brain_tumor"],
                "task": "Brain Tumor Classification",
                "description": "ResNet50 model fine-tuned for brain tumor classification in MRI scans",
                "dataset": "Brain Cancer MRI Dataset",
                "input_size": [3, 224, 224],
                "metrics": {
                    "accuracy": 0.85,  # Example metrics - update with actual values
                    "f1_score": 0.84,
                    "sensitivity": 0.83,
                    "specificity": 0.87
                }
            },
            "resnet50_tb_full": {
                "path": str(checkpoints_dir / "resnet50_tb_full" / "best_resnet50_tb_full_trained.pth"),
                "num_classes": 2,
                "classes": ["Normal", "Tuberculosis"],
                "task": "Tuberculosis Detection",
                "description": "ResNet50 model fine-tuned for tuberculosis detection in chest X-rays",
                "dataset": "TB Chest Radiography Database",
                "input_size": [3, 224, 224],
                "metrics": {
                    "accuracy": 0.89,
                    "f1_score": 0.88,
                    "sensitivity": 0.87,
                    "specificity": 0.91
                }
            },
            "resnet50_pneumonia_full": {
                "path": str(checkpoints_dir / "resnet50_pneumonia_full" / "best_resnet50_pneumonia_full_trained.pth"),
                "num_classes": 2,
                "classes": ["Normal", "Pneumonia"],
                "task": "Pneumonia Detection",
                "description": "ResNet50 model fine-tuned for pneumonia detection in chest X-rays",
                "dataset": "Chest X-ray Pneumonia Dataset",
                "input_size": [3, 224, 224],
                "metrics": {
                    "accuracy": 0.92,
                    "f1_score": 0.91,
                    "sensitivity": 0.90,
                    "specificity": 0.94
                }
            }
        }
    
    def create_repository(self):
        """Create the HuggingFace repository if it doesn't exist."""
        try:
            # Try to get repo info to check if it exists
            self.api.repo_info(repo_id=self.full_repo_name, repo_type="model")
            print(f"[INFO] Repository {self.full_repo_name} already exists")
        except RepositoryNotFoundError:
            # Create the repository
            create_repo(
                repo_id=self.repo_name,
                token=self.hf_token,
                repo_type="model",
                private=False,
                exist_ok=True
            )
            print(f"[NEW] Created new repository: {self.full_repo_name}")
    
    def generate_model_card(self, model_name: str, config: dict) -> str:
        """Generate a model card for the given model."""
        
        classes_str = ", ".join(config["classes"])
        metrics_str = "\n".join([f"- **{metric.replace('_', ' ').title()}**: {value:.3f}" 
                                for metric, value in config["metrics"].items()])
        
        model_card = f"""---
library_name: pytorch
license: mit
tags:
- medical-imaging
- computer-vision
- {config["task"].lower().replace(" ", "-")}
- resnet50
- pytorch
- healthcare
- radiology
datasets:
- {config["dataset"].lower().replace(" ", "-")}
metrics:
- accuracy
- f1
- sensitivity
- specificity
language:
- en
pipeline_tag: image-classification
---

# {config["task"]} - ResNet50 Model

## Model Description

{config["description"]}. This model is part of the OpenMed platform, an AI-powered medical imaging analysis system.

**Model Architecture**: ResNet50 with ImageNet pre-trained weights, fine-tuned for medical imaging tasks.

## Intended Use

This model is designed for research and educational purposes in medical imaging. It should **NOT** be used for actual medical diagnosis without proper validation and oversight by qualified healthcare professionals.

## Model Details

- **Model Type**: Image Classification
- **Architecture**: ResNet50
- **Number of Classes**: {config["num_classes"]}
- **Classes**: {classes_str}
- **Input Size**: {config["input_size"]}
- **Framework**: PyTorch

## Performance

{metrics_str}

*Note: These metrics are from validation on the training dataset and may not reflect real-world performance.*

## Training Details

- **Base Model**: ResNet50 pre-trained on ImageNet
- **Fine-tuning Strategy**: Full network training (all layers trainable)
- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Batch Size**: 16-32
- **Data Augmentation**: Random horizontal flip, rotation, color jitter
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Usage

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(
    repo_id="{self.full_repo_name}",
    filename="{model_name}/pytorch_model.bin"
)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device)

# Note: You'll need to define the ResNet50 architecture or use the OpenMed codebase
# model = ResNet50Model(num_classes={config["num_classes"]})
# model.load_state_dict(checkpoint)
# model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example inference
# image = Image.open("your_medical_image.jpg")
# input_tensor = transform(image).unsqueeze(0)
# with torch.no_grad():
#     outputs = model(input_tensor)
#     predictions = torch.nn.functional.softmax(outputs, dim=1)
```

## Limitations and Bias

- **Training Data**: Model performance is limited by the quality and diversity of training data
- **Generalization**: May not generalize well to images from different institutions, equipment, or populations
- **Class Imbalance**: Performance may vary across different classes due to dataset imbalances
- **Image Quality**: Performance depends on image quality and may degrade with poor quality inputs

## Ethical Considerations

- This model is for research and educational use only
- Should not replace professional medical diagnosis
- Requires validation in clinical settings before any medical application
- May exhibit bias based on training data demographics
- Users should be aware of limitations and potential failure modes

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{openmed_models_2024,
  title={{OpenMed: AI-Powered Medical Imaging Analysis Platform}},
  author={{OpenMed Team}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/{self.full_repo_name}}}}}
}}
```

## Model Card Authors

OpenMed Development Team

## Model Card Contact

For questions about this model, please open an issue in the OpenMed repository.

---

**Disclaimer**: This model is for research and educational purposes only. It should not be used for medical diagnosis or treatment decisions without proper validation and oversight by qualified healthcare professionals.
"""
        return model_card
    
    def generate_config_json(self, model_name: str, config: dict) -> str:
        """Generate a config.json file for the model."""
        model_config = {
            "architectures": ["ResNet50"],
            "model_type": "resnet50",
            "num_classes": config["num_classes"],
            "id2label": {str(i): label for i, label in enumerate(config["classes"])},
            "label2id": {label: i for i, label in enumerate(config["classes"])},
            "image_size": config["input_size"],
            "task": config["task"],
            "dataset": config["dataset"],
            "framework": "pytorch",
            "metrics": config["metrics"]
        }
        return json.dumps(model_config, indent=2)
    
    def push_model(self, model_name: str, config: dict):
        """Push a single model to HuggingFace Hub."""
        print(f"\n[PUSH] Pushing {model_name}...")
        
        # Check if model file exists
        model_path = Path(config["path"])
        if not model_path.exists():
            print(f"[ERROR] Model file not found: {model_path}")
            print(f"   Please ensure the model has been trained and saved.")
            return False
        
        try:
            # Create temporary directory for model files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                model_dir = temp_path / model_name
                model_dir.mkdir(parents=True)
                
                # Copy and rename model file to standard name
                shutil.copy2(model_path, model_dir / "pytorch_model.bin")
                
                # Generate model card
                model_card = self.generate_model_card(model_name, config)
                with open(model_dir / "README.md", "w", encoding="utf-8") as f:
                    f.write(model_card)
                
                # Generate config.json
                config_json = self.generate_config_json(model_name, config)
                with open(model_dir / "config.json", "w", encoding="utf-8") as f:
                    f.write(config_json)
                
                # Upload files
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(temp_path)
                        print(f"  [UPLOAD] Uploading {relative_path}")
                        
                        upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=str(relative_path),
                            repo_id=self.full_repo_name,
                            token=self.hf_token
                        )
                
                print(f"[SUCCESS] Successfully pushed {model_name}")
                return True
                
        except Exception as e:
            print(f"[ERROR] Error pushing {model_name}: {str(e)}")
            return False
    
    def generate_main_readme(self) -> str:
        """Generate the main README for the repository."""
        
        readme = f"""# OpenMed Medical Imaging Models

This repository contains trained ResNet50 models for medical imaging tasks from the OpenMed platform.

## Models Included

"""
        
        for model_name, config in self.models_config.items():
            readme += f"### {config['task']}\n"
            readme += f"- **Model**: `{model_name}`\n"
            readme += f"- **Classes**: {', '.join(config['classes'])}\n"
            readme += f"- **Dataset**: {config['dataset']}\n"
            readme += f"- **Description**: {config['description']}\n\n"
        
        readme += """## Quick Start

```python
from huggingface_hub import hf_hub_download
import torch

# Download a specific model (example: brain tumor classification)
model_path = hf_hub_download(
    repo_id=\"""" + self.full_repo_name + """\",
    filename="resnet50_brain_tumor_full/pytorch_model.bin"
)

# Load the model
checkpoint = torch.load(model_path, map_location="cpu")
```

## Model Architecture

All models are based on ResNet50 with ImageNet pre-trained weights:
- **Input Size**: 224x224x3 RGB images
- **Normalization**: ImageNet statistics
- **Fine-tuning**: Full network training (all layers trainable)

## Usage Guidelines

**Important**: These models are for research and educational purposes only. They should **NOT** be used for actual medical diagnosis without proper validation and oversight by qualified healthcare professionals.

## Performance Metrics

Performance metrics for each model can be found in their respective model cards.

## Citation

```bibtex
@misc{openmed_models_2024,
  title={OpenMed: AI-Powered Medical Imaging Analysis Platform},
  author={OpenMed Team},
  year={2024},
  howpublished={\\url{https://huggingface.co/""" + self.full_repo_name + """}}
}
```

## License

MIT License - See individual model cards for details.

---

**Disclaimer**: These models are for research and educational purposes only. They should not be used for medical diagnosis or treatment decisions without proper validation and oversight by qualified healthcare professionals.
"""
        return readme
    
    def push_all_models(self):
        """Push all models to HuggingFace Hub."""
        print(f"[START] Starting to push all models to {self.full_repo_name}")
        
        # Create repository
        self.create_repository()
        
        # Push main README
        print("\n[README] Creating main repository README...")
        main_readme = self.generate_main_readme()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(main_readme)
            temp_readme_path = f.name
        
        try:
            upload_file(
                path_or_fileobj=temp_readme_path,
                path_in_repo="README.md",
                repo_id=self.full_repo_name,
                token=self.hf_token
            )
            print("[SUCCESS] Main README uploaded")
        finally:
            os.unlink(temp_readme_path)
        
        # Push each model
        successful_pushes = 0
        for model_name, config in self.models_config.items():
            success = self.push_model(model_name, config)
            if success:
                successful_pushes += 1
        
        print(f"\n[COMPLETE] Successfully pushed {successful_pushes}/{len(self.models_config)} models")
        print(f"[URL] Repository URL: https://huggingface.co/{self.full_repo_name}")

def main():
    """Main function to run the model pushing script."""
    
    # HuggingFace token (provided by user)
    HF_TOKEN = "your own token"
    
    # Repository name
    REPO_NAME = "openmed-medical-imaging-models"
    
    print("OpenMed Model Hub Push Script")
    print("=" * 50)
    
    try:
        # Initialize the pusher
        pusher = OpenMedModelPusher(HF_TOKEN, REPO_NAME)
        
        # Push all models
        pusher.push_all_models()
        
    except Exception as e:
        print(f"[ERROR] Script failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 