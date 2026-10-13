"""
train.py - Training Pipeline for Handwritten Digit Recognition ANN.

This script executes the complete training pipeline:
    1. Loads the official MNIST dataset using `tf.keras.datasets.mnist.load_data()`.
    2. Normalizes pixel values from [0, 255] to [0.0, 1.0].
    3. Flattens 28x28 images into 784-dimensional feature vectors.
    4. Builds and compiles the Artificial Neural Network (from model.py).
    5. Trains the model (5 epochs, batch size 32, 10% validation split).
    6. Evaluates performance on the 10,000-sample test set.
    7. Saves the trained model to `models/mnist_ann_model.keras`.
    8. Saves training history to `outputs/training_history.json` and plots training curves.

Usage:
    python train.py
    python train.py --force          # Retrain even if a saved model already exists
    python train.py --epochs 10      # Custom epoch count
"""

import os
import sys
import json
import argparse
from typing import Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Local imports
from model import build_ann_model


# Standard paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(MODELS_DIR, "mnist_ann_model.keras")
HISTORY_PATH = os.path.join(OUTPUTS_DIR, "training_history.json")
CURVES_PLOT_PATH = os.path.join(OUTPUTS_DIR, "training_curves.png")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across NumPy and TensorFlow for reproducible results."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_and_preprocess_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST handwritten digit dataset and prepares it for ANN input.

    Returns:
        Tuple: (X_train, y_train, X_test, y_test)
            - X_train: Shape (60000, 784), float32 normalized to [0.0, 1.0].
            - y_train: Shape (60000,), integer labels 0-9.
            - X_test: Shape (10000, 784), float32 normalized to [0.0, 1.0].
            - y_test: Shape (10000,), integer labels 0-9.
    """
    print("[1/5] Loading MNIST dataset via tf.keras.datasets.mnist...")
    # MNIST is cached locally in ~/.keras/datasets/mnist.npz after the first download
    (X_train_raw, y_train), (X_test_raw, y_test) = tf.keras.datasets.mnist.load_data()

    print(f"      Raw Train images: {X_train_raw.shape}, Labels: {y_train.shape}")
    print(f"      Raw Test images:  {X_test_raw.shape}, Labels: {y_test.shape}")

    print("[2/5] Normalizing pixel intensities to [0.0, 1.0] and flattening to 784 features...")
    # Normalize: Convert uint8 [0, 255] to float32 [0.0, 1.0]
    X_train = X_train_raw.astype(np.float32) / 255.0
    X_test = X_test_raw.astype(np.float32) / 255.0

    # Flatten: Reshape (N, 28, 28) into (N, 784)
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    print(f"      Processed Train features: {X_train.shape} (min={X_train.min():.1f}, max={X_train.max():.1f})")
    print(f"      Processed Test features:  {X_test.shape} (min={X_test.min():.1f}, max={X_test.max():.1f})")

    return X_train, y_train, X_test, y_test


def plot_and_save_training_curves(history: Dict[str, Any], output_path: str) -> None:
    """
    Generates and saves the loss and accuracy training vs validation curves.

    Args:
        history: Dictionary containing loss, accuracy, val_loss, val_accuracy.
        output_path: File path to save the generated PNG plot.
    """
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ANN Model Training History (MNIST)", fontsize=14, fontweight="bold")

    # Accuracy Plot
    ax1.plot(epochs, history["accuracy"], "o-", label="Training Accuracy", color="#2563EB", linewidth=2)
    ax1.plot(epochs, history["val_accuracy"], "s--", label="Validation Accuracy", color="#10B981", linewidth=2)
    ax1.set_title("Model Accuracy Across Epochs", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim([0.85, 1.01])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax1_max_loss = max(max(history["loss"]), max(history["val_loss"]))
    ax2.plot(epochs, history["loss"], "o-", label="Training Loss", color="#DC2626", linewidth=2)
    ax2.plot(epochs, history["val_loss"], "s--", label="Validation Loss", color="#F59E0B", linewidth=2)
    ax2.set_title("Model Loss Across Epochs", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Sparse Categorical Cross-Entropy", fontsize=11)
    ax2.set_ylim([0.0, ax1_max_loss * 1.15])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"      Saved training curves plot to: {output_path}")


def train_ann(
    epochs: int = 5,
    batch_size: int = 32,
    validation_split: float = 0.10,
    force_retrain: bool = False,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Executes the training workflow.

    Args:
        epochs: Number of complete passes over the training set (default: 5).
        batch_size: Number of samples per gradient update (default: 32).
        validation_split: Fraction of training data reserved for validation (default: 0.10).
        force_retrain: If True, retrains even if a saved model already exists.

    Returns:
        Tuple of (trained_model, history_dict).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Check for existing model
    if os.path.exists(MODEL_PATH) and not force_retrain:
        print(f"Found existing trained model at: {MODEL_PATH}")
        print("Use '--force' flag or pass force_retrain=True to train a new model from scratch.")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            history = {}
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                    history = json.load(f)
            print("Loaded existing model successfully.")
            return model, history
        except Exception as e:
            print(f"Warning: Failed to load existing model ({e}). Retraining from scratch...")

    # Set seeds for reproducibility
    set_seed(42)

    # 1 & 2. Load and preprocess MNIST dataset
    X_train, y_train, X_test, y_test = load_and_preprocess_mnist()

    # 3 & 4. Build and compile ANN
    print("\n[3/5] Constructing Artificial Neural Network architecture...")
    model = build_ann_model(input_shape=(784,), learning_rate=0.001)
    model.summary()

    # 5. Train model
    print(f"\n[4/5] Training ANN for {epochs} epochs (batch_size={batch_size}, val_split={validation_split:.0%})...")
    history_obj = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    history = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}

    # 6 & 7. Evaluate on official test set
    print("\n[5/5] Evaluating trained model on official MNIST test set (10,000 samples)...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("=" * 60)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
    print("=" * 60)

    # 8. Save trained model
    print(f"\nSaving trained model to: {MODEL_PATH}")
    model.save(MODEL_PATH)

    # 9. Save history and curves
    history["test_loss"] = float(test_loss)
    history["test_accuracy"] = float(test_accuracy)
    history["epochs_trained"] = epochs
    history["batch_size"] = batch_size

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {HISTORY_PATH}")

    plot_and_save_training_curves(history, CURVES_PLOT_PATH)

    print("\nTraining and artifact generation complete!")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST Handwritten Digit ANN Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--force", action="store_true", help="Force retrain even if a saved model exists")
    args = parser.parse_args()

    train_ann(
        epochs=args.epochs,
        batch_size=args.batch_size,
        force_retrain=args.force,
    )
