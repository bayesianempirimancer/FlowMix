import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import checkpoints
import numpy as np
import os

class Evaluator:
    def __init__(self, model, test_data, metrics):
        self.model = model
        self.test_data = test_data
        self.metrics = metrics

    def evaluate(self, params):
        total_loss = 0.0
        total_metrics = {metric: 0.0 for metric in self.metrics}

        for batch in self.test_data:
            y, h = batch['y'], batch['h']
            reconstructed, loss = self.model.apply(params, y, h)
            total_loss += loss

            for metric in self.metrics:
                total_metrics[metric] += self.compute_metric(metric, batch, reconstructed)

        avg_loss = total_loss / len(self.test_data)
        avg_metrics = {metric: total_metrics[metric] / len(self.test_data) for metric in self.metrics}

        return avg_loss, avg_metrics

    def compute_metric(self, metric, batch, reconstructed):
        if metric == 'mse':
            return jnp.mean((batch['points'] - reconstructed) ** 2)
        elif metric == 'mae':
            return jnp.mean(jnp.abs(batch['points'] - reconstructed))
        # Add more metrics as needed
        return 0.0

def load_model(checkpoint_path):
    if os.path.exists(checkpoint_path):
        return checkpoints.restore_checkpoint(checkpoint_path, target=None)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

def main():
    # Load your model and data here
    model = ...  # Load your VAE model
    test_data = ...  # Load your test dataset
    metrics = ['mse', 'mae']  # Define the metrics you want to compute

    evaluator = Evaluator(model, test_data, metrics)
    params = load_model('path/to/checkpoint')  # Specify your checkpoint path

    avg_loss, avg_metrics = evaluator.evaluate(params)
    print(f"Average Loss: {avg_loss}")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()