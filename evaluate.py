"""
evaluate.py - Evaluation Pipeline for Trained Handwritten Digit ANN.

This module evaluates the trained ANN model strictly on the official MNIST
test dataset (10,000 samples).

Metrics and Visualizations Generated:
    - Test Accuracy & Test Loss (Sparse Categorical Cross-Entropy)
    - Classification Report: Precision, Recall, and F1-Score per class (0-9)
    - Confusion Matrix (saved to `outputs/confusion_matrix.png`)
    - Sample Predictions Grid (saved to `outputs/sample_predictions.png`)
    - Evaluation Summary JSON (saved to `outputs/evaluation_metrics.json`)

Usage:
    python evaluate.py
"""

import os
import sys
import json
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Standard paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(MODELS_DIR, "mnist_ann_model.keras")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "evaluation_metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
SAMPLE_PREDICTIONS_PATH = os.path.join(OUTPUTS_DIR, "sample_predictions.png")


def load_model_safely(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Loads the trained Keras model, providing clear error guidance if absent.

    Args:
        model_path: Path to the .keras model file.

    Returns:
        Loaded tf.keras.Model instance.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'.\n"
            f"Please run the training script first:\n"
            f"    python train.py"
        )
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model file '{model_path}': {e}\n"
            f"The file might be corrupted or incompatible. Try retraining:\n"
            f"    python train.py --force"
        )


def load_test_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and prepares the official MNIST test partition (10,000 images).

    Returns:
        Tuple: (X_test_flat, y_test, X_test_2d)
    """
    # Load raw MNIST
    (_, _), (X_test_raw, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0.0, 1.0]
    X_test_norm = X_test_raw.astype(np.float32) / 255.0

    # Flatten for ANN input
    X_test_flat = X_test_norm.reshape(-1, 784)

    return X_test_flat, y_test, X_test_raw


def plot_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    """
    Generates and saves a high-resolution confusion matrix heatmap.

    Args:
        cm: 10x10 numpy array containing confusion counts.
        output_path: File path to save the PNG image.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Sample Count", rotation=-90, va="bottom", fontsize=11)

    classes = [str(i) for i in range(10)]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="ANN Confusion Matrix (MNIST Official Test Set)",
        ylabel="True Digit Label",
        xlabel="Predicted Digit Label",
    )
    ax.set_title("ANN Confusion Matrix (MNIST Official Test Set)", fontsize=13, fontweight="bold", pad=12)

    # Threshold for text contrast
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"      Saved confusion matrix plot to: {output_path}")


def plot_sample_predictions(
    images_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    output_path: str,
    num_samples: int = 15,
) -> None:
    """
    Visualizes a grid of sample test predictions with true vs predicted labels.

    Args:
        images_raw: Raw 28x28 grayscale images.
        y_true: True ground-truth labels.
        y_pred: Predicted class labels.
        probabilities: Softmax probability arrays.
        output_path: File path to save the PNG image.
        num_samples: Total number of samples to visualize.
    """
    rows = 3
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(13, 8))
    fig.suptitle("Sample Predictions on MNIST Test Partition", fontsize=14, fontweight="bold")

    np.random.seed(42)
    sample_indices = np.random.choice(len(y_true), size=num_samples, replace=False)

    for idx, ax in zip(sample_indices, axes.flat):
        img = images_raw[idx]
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        conf = probabilities[idx][pred_label] * 100.0

        is_correct = true_label == pred_label
        color = "#15803D" if is_correct else "#DC2626"  # Green vs Red

        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"True: {true_label} | Pred: {pred_label}\nConf: {conf:.1f}%",
            color=color,
            fontsize=10,
            fontweight="bold" if not is_correct else "normal",
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"      Saved sample predictions grid to: {output_path}")


def evaluate_model() -> Dict[str, Any]:
    """
    Runs the full evaluation procedure and generates metrics and figures.

    Returns:
        Dictionary containing all computed metric values.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 65)
    print("      ANN MODEL EVALUATION (OFFICIAL MNIST TEST PARTITION)      ")
    print("=" * 65)

    print("[1/4] Loading trained model from disk...")
    model = load_model_safely(MODEL_PATH)
    print("      Model loaded successfully.")

    print("[2/4] Loading 10,000 official test samples...")
    X_test_flat, y_test, X_test_raw = load_test_dataset()

    print("[3/4] Computing test set predictions and loss...")
    test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
    probabilities = model.predict(X_test_flat, batch_size=64, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    print("\n[4/4] Computing classification metrics...")
    cm = confusion_matrix(y_test, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    report_str = classification_report(y_test, y_pred, digits=4)
    print("\n" + report_str)

    # Compile structured metrics dictionary
    per_class_metrics = {}
    for digit in range(10):
        per_class_metrics[str(digit)] = {
            "precision": float(prec[digit]),
            "recall": float(rec[digit]),
            "f1_score": float(f1[digit]),
            "support": int(support[digit]),
        }

    metrics = {
        "dataset": "MNIST Test Set (10,000 samples)",
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1_score": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1_score": float(weighted_f1),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }

    # Save metrics JSON
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics JSON to: {METRICS_PATH}")

    # Generate plots
    plot_confusion_matrix(cm, CONFUSION_MATRIX_PATH)
    plot_sample_predictions(X_test_raw, y_test, y_pred, probabilities, SAMPLE_PREDICTIONS_PATH)

    print("\n" + "=" * 65)
    print(f"  Summary Evaluation Results:")
    print(f"  - Test Loss (Sparse Categorical Cross-Entropy): {test_loss:.4f}")
    print(f"  - Test Accuracy:                               {test_acc * 100:.2f}%")
    print(f"  - Macro Precision:                             {macro_prec * 100:.2f}%")
    print(f"  - Macro Recall:                                {macro_rec * 100:.2f}%")
    print(f"  - Macro F1-Score:                              {macro_f1 * 100:.2f}%")
    print("=" * 65)

    return metrics


if __name__ == "__main__":
    evaluate_model()
