# Enhanced-Super-Resolution GAN (ESRGAN) Project

<img src="https://raw.githubusercontent.com/atikul-islam-sajib/Research-Assistant-Work-/main/IMG_9292.jpg" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">

This project provides a complete framework for training and testing a Enhanced Super-Resolution Generative Adversarial Network (SR-GAN). It includes functionality for data preparation, model training, testing, and inference to enhance low-resolution images to high-resolution.

<img src="https://esrgan.readthedocs.io/en/latest/_images/architecture.png" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">

## Features

| Feature                          | Description                                                                                                                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Efficient Implementation**     | Utilizes an optimized SRGAN model architecture for superior performance on diverse image segmentation tasks.                                                                                                          |
| **Custom Dataset Support**       | Features easy-to-use data loading utilities that seamlessly accommodate custom datasets, requiring minimal configuration.                                                                                             |
| **Training and Testing Scripts** | Provides streamlined scripts for both training and testing phases, simplifying the end-to-end workflow.                                                                                                               |
| **Visualization Tools**          | Equipped with tools for tracking training progress and visualizing segmentation outcomes, enabling clear insight into model effectiveness.                                                                            |
| **Custom Training via CLI**      | Offers a versatile command-line interface for personalized training configurations, enhancing flexibility in model training.                                                                                          |
| **Import Modules**               | Supports straightforward integration into various projects or workflows with well-documented Python modules, simplifying the adoption of SRGAN functionality.                                                         |
| **Multi-Platform Support**       | Guarantees compatibility with various computational backends, including MPS for GPU acceleration on Apple devices, CPU, and CUDA for Nvidia GPU acceleration, ensuring adaptability across different hardware setups. |

## Getting Started

## Installation Instructions

Follow these steps to get the project set up on your local machine:

| Step | Instruction                                  | Command                                                       |
| ---- | -------------------------------------------- | ------------------------------------------------------------- |
| 1    | Clone this repository to your local machine. | **git clone https://github.com/atikul-islam-sajib/ESRGAN.git** |
| 2    | Navigate into the project directory.         | **cd ESRGAN**                                                  |
| 3    | Install the required Python packages.        | **pip install -r requirements.txt**                           |

## Project Structure

This project is thoughtfully organized to support the development, training, and evaluation of the srgan model efficiently. Below is a concise overview of the directory structure and their specific roles:

- **checkpoints/**
  - Stores model checkpoints during training for later resumption.
- **best_model/**

  - Contains the best-performing model checkpoints as determined by validation metrics.

- **train_models/**

  - Houses all model checkpoints generated throughout the training process.

- **data/**

  - **processed/**: Processed data ready for modeling, having undergone normalization, augmentation, or encoding.
  - **raw/**: Original, unmodified data serving as the baseline for all preprocessing.

- **logs/**

  - **Log** files for debugging and tracking model training progress.

- **metrics/**

  - Files related to model performance metrics for evaluation purposes.

- **outputs/**

  - **test_images/**: Images generated during the testing phase, including segmentation outputs.
  - **train_gif/**: GIFs compiled from training images showcasing the model's learning progress.
  - **train_images/**: Images generated during training for performance visualization.

- **research/**

  - **notebooks/**: Jupyter notebooks for research, experiments, and exploratory analyses conducted during the project.

- **src/**

  - Source code directory containing all custom modules, scripts, and utility functions for the U-Net model.

- **unittest/**
  - Unit tests ensuring code reliability, correctness, and functionality across various project components.

### Dataset Organization for ESRGAN

The dataset is organized into three categories for ESRGAN. Each category directly contains paired images and their corresponding lower resolution images and higher resolution, stored together to simplify the association between lower resolution and higher resolution images .

## Directory Structure:

```
dataset/
├── HR/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
├── LR/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
```

### User Guide Notebook - CLI

For detailed documentation on the implementation and usage, visit the -> [ESRGAN Notebook - CLI](https://github.com/atikul-islam-sajib/ESRGAN/blob/main/research/notebooks/ESRGAN_CLI.ipynb).

### User Guide Notebook - Custom Modules

For detailed documentation on the implementation and usage, visit the -> [ESRGAN Notebook - CM](https://github.com/atikul-islam-sajib/ESRGAN/blob/main/research/notebooks/ESRGAN_CM.ipynb).

## Data Versioning with DVC
To ensure you have the correct version of the dataset and model artifacts.

Reproducing the Pipeline
To reproduce the entire pipeline and ensure consistency, use:

```bash
dvc repro
```

### Command Line Interface

The project is controlled via a command line interface (CLI) which allows for running different operational modes such as training, testing, and inference.

#### CLI Arguments
| Argument          | Description                                  | Type   | Default |
|-------------------|----------------------------------------------|--------|---------|
| `--image_path`    | Path to the image dataset                    | str    | None    |
| `--batch_size`    | Number of images per batch                   | int    | 1       |
| `--image_size`    | Size to resize images to                     | int    | 64      |
| `--epochs`        | Number of training epochs                    | int    | 100     |
| `--lr`            | Learning rate                                | float  | 0.0002  |
| `--content_loss`  | Weight of content loss                       | float  | 0.001   |
| `--lr_scheduler`| Enable learning rate scheduler              | bool   | False   |
| `--is_weight_init`| Apply weight initialization                  | bool   | False   |
| `--verbose`    | Display detailed loss information            | bool   | False   |
| `--device`        | Computation device ('cuda', 'mps', 'cpu')    | str    | 'mps'   |
| `--adam`          | Use Adam optimizer                           | bool   | True    |
| `--SGD`           | Use Stochastic Gradient Descent optimizer    | bool   | False   |
| `--beta1`         | Beta1 parameter for Adam optimizer           | float  | 0.5     |
| `--train`         | Flag to initiate training mode               | action | N/A     |
| `--model`         | Path to a saved model for testing            | str    | None    |
| `--test`          | Flag to initiate testing mode                | action | N/A     |

### CLI Command Examples

| Task                     | CUDA Command                                                                                                              | MPS Command                                                                                                              | CPU Command                                                                                                              |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **Training a Model**     | `python cli.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --content_loss 0.01 --adam True --is_l1 True --device "cuda"` | `python cli.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --content_loss 0.01 --adam True --is_l1 True --device "mps"` | `python main.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --content_loss 0.01 --adam True --is_l1 True --device "cpu"` |
| **Testing a Model**      | `python cli.py --test --model "/path/to/saved_model.pth" --device "cuda"`                                              | `python cli.py --test --model "/path/to/saved_model.pth" --device "mps"`                                              | `python cli.py --test --model "/path/to/saved_model.pth" --device "cpu"`                                              |

### Notes:
- **CUDA Command**: For systems with NVIDIA GPUs, using the `cuda` device will leverage GPU acceleration.
- **MPS Command**: For Apple Silicon (M1, M2 chips), using the `mps` device can provide optimized performance.
- **CPU Command**: Suitable for systems without dedicated GPU support or for testing purposes on any machine.

**Configure the Project**:
   Update the `config.yml` file with the appropriate paths and settings. An example `config.yml`:
   ```yaml
  path: 
    RAW_DATA_PATH: "./data/raw/"
    PROCESSED_DATA_PATH: "./data/processed/"
    ARTIFACTS_PATH: "./research/artifacts/"
    TRAIN_IMAGES_PATH: "./outputs/train_images/"
    TEST_IMAGE_PATH: "./outputs/test_images/"
    TRAIN_MODEL_CHECKPOINT_PATH: "./checkpoints/train_models"
    METRICS_PATH: "./outputs/metrics/"
    BEST_MODEL_CHECKPOINT_PATH: "./checkpoints/best_model"

  dataloader:
    image_path: "./data/raw/dataset.zip"
    image_size: 64
    split_size: 0.40
    batch_size: 4

  trainer:
    epochs: 500
    lr: 0.0002
    beta1: 0.5
    beta2: 0.999
    adam: True
    SGD: False
    momentum: 0.9
    content_loss: 0.01
    pixel_loss: 0.05
    device: "mps"
    lr_scheduler: False
    weight_init: True

  tester:
    model: "best"       # "./checkpoints/best_model/netG.pth"
    device: "mps"
    dataloader: "valid"
```


#### Initializing Data Loader - Custom Modules
```python
loader = Loader(image_path="path/to/dataset", batch_size=32, image_size=128)
loader.unzip_folder()
loader.create_dataloader()
```

##### To details about dataset
```python
loader.plot_images()           # It will display the images from dataset
```

#### Training the Model
```python
trainer = Trainer(
    epochs=100,                # Number of epochs to train the model
    lr=0.0002,                 # Learning rate for optimizer
    content_loss=0.1,          # Weight for content loss in the loss calculation
    device='cuda',             # Computation device ('cuda', 'mps', 'cpu')
    adam=True,                 # Use Adam optimizer; set to False to use SGD if implemented
    SGD=False,                 # Use Stochastic Gradient Descent optimizer; typically False if Adam is True
    beta1=0.5,                 # Beta1 parameter for Adam optimizer
    lr_scheduler=False,        # Enable a learning rate scheduler
    weight_init=False,         # Enable custom weight initialization for the models
    verbose=True               # Display training progress and statistics
)

# Start training
trainer.train()
```

##### Training Performances
```python
print(trainer.plot_history())    # It will plot the netD and netG losses for each epochs
```

#### Testing the Model
```python
tester = Tester(device="cuda", model="path/to/model.pth") # use mps, cpu
test.test()
```

### Configuration for MLFlow

1. **Generate a Personal Access Token on DagsHub**:
   - Log in to [DagsHub](https://dagshub.com).
   - Go to your user settings and generate a new personal access token under "Personal Access Tokens".

2. **Set environment variables**:
   Set the following environment variables with your DagsHub credentials:
   ```bash
   export MLFLOW_TRACKING_URI="https://dagshub.com/<username>/<repo_name>.mlflow"
   export MLFLOW_TRACKING_USERNAME="<your_dagshub_username>"
   export MLFLOW_TRACKING_PASSWORD="<your_dagshub_token>"
   ```

   Replace `<username>`, `<repo_name>`, `<your_dagshub_username>`, and `<your_dagshub_token>` with your actual DagsHub username, repository name, and personal access token.

### Running the Training Script

To start training and logging the experiments to DagsHub, run the following command:

```bash
python src/train.py
```

### Accessing Experiment Tracking

You can access the MLflow experiment tracking UI hosted on DagsHub using the following link:

[ESRGAN Experiment Tracking on DagsHub](https://dagshub.com/atikul-islam-sajib/ESRGAN/experiments/#/)

### Using MLflow UI Locally

If you prefer to run the MLflow UI locally, use the following command:

```bash
mlflow ui
```


## Contributing
Contributions to improve this implementation of ESRGAN are welcome. Please follow the standard fork-branch-pull request workflow.

## License
Specify the license under which the project is made available (e.g., MIT License).
