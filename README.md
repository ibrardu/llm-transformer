# llm-transformer (nanoGPT-custom)

This repository contains code for training a custom Transformer model based on the [nanoGPT](https://github.com/karpathy/nanoGPT) implementation. The setup includes configuration, model definition, data handling, and training scripts.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/nanoGPT-custom.git
    cd nanoGPT-custom
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Add your dataset to the `data` directory.

## Training

To train the model, you can use the provided notebook or the training script.

### Using the Notebook

Open `notebooks/train_model.ipynb` in your preferred Jupyter environment and follow the steps to train the model.

### Using the Script

Run the training script:
```bash
python scripts/train.py
