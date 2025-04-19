# Flax VAE Project

This project implements a Variational Autoencoder (VAE) for object-centered embedding of 3D point cloud data using Flax and JAX. The model is designed to process a set of 3D locations and associated feature vectors, producing a compact embedding representation.

## Project Structure

```
flax_vae_project
├── src
│   ├── models
│   │   ├── vae.py         # Main VAE class integrating encoder and decoder
│   │   ├── encoder.py     # Implementation of the encoder model
│   │   └── decoder.py     # Implementation of the decoder model
│   ├── layers
│   │   ├── attention.py    # Self-attention and cross-attention layers
│   │   └── pooling.py      # Distance-based pooling layers
│   ├── utils
│   │   └── data_utils.py   # Utility functions for data loading and preprocessing
│   ├── train.py           # Training loop setup for the VAE
│   └── eval.py            # Evaluation functions for the trained model
├── config
│   └── default.yaml       # Default configuration settings
├── README.md              # Project documentation
└── requirements.txt       # Required Python packages
```

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd flax_vae_project
pip install -r requirements.txt
```

## Usage

1. Prepare your 3D point cloud data in the expected format.
2. Modify the `config/default.yaml` file to set your hyperparameters and data paths.
3. Run the training script:

```bash
python src/train.py
```

4. After training, evaluate the model using:

```bash
python src/eval.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.