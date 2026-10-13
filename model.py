"""
model.py - Artificial Neural Network (ANN) Architecture Definition.

This module defines the fully connected Multilayer Perceptron (MLP) architecture
for handwritten digit recognition (classes 0-9) using TensorFlow and Keras.

Architecture Specifications:
    - Input: 784 features (corresponding to flattened 28x28 grayscale images)
    - Hidden Layer 1: 128 neurons, ReLU activation function
    - Hidden Layer 2: 64 neurons, ReLU activation function
    - Output Layer: 10 neurons, Softmax activation function (probabilities for digits 0-9)

Compiler Configuration:
    - Optimizer: Adam (Adaptive Moment Estimation)
    - Loss: Sparse Categorical Cross-Entropy (integer class labels)
    - Evaluation Metric: Accuracy
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_ann_model(
    input_shape: tuple = (784,),
    learning_rate: float = 0.001,
) -> models.Sequential:
    """
    Constructs and compiles the fully connected Artificial Neural Network.

    Mathematical Formulation:
        1. Hidden Layer 1:
           z_1 = W_1 * x + b_1
           a_1 = ReLU(z_1) = max(0, z_1)
           Where W_1 in R^(128 x 784), b_1 in R^128

        2. Hidden Layer 2:
           z_2 = W_2 * a_1 + b_2
           a_2 = ReLU(z_2) = max(0, z_2)
           Where W_2 in R^(64 x 128), b_2 in R^64

        3. Output Layer:
           z_3 = W_3 * a_2 + b_3
           p_i = Softmax(z_3)_i = exp(z_3,i) / sum_j exp(z_3,j)
           Where W_3 in R^(10 x 64), b_3 in R^10

    Args:
        input_shape: Tuple representing feature dimensions (default: (784,)).
        learning_rate: Initial learning rate for the Adam optimizer (default: 0.001).

    Returns:
        Compiled tf.keras.models.Sequential instance.
    """
    model = models.Sequential(
        [
            # Explicit Input specification (784 normalized grayscale pixel values)
            layers.Input(shape=input_shape, name="input_features"),
            # First hidden fully connected layer: 128 neurons with ReLU activation
            layers.Dense(
                units=128,
                activation="relu",
                name="dense_hidden_1",
            ),
            # Second hidden fully connected layer: 64 neurons with ReLU activation
            layers.Dense(
                units=64,
                activation="relu",
                name="dense_hidden_2",
            ),
            # Output layer: 10 neurons representing classes 0 through 9 with Softmax
            layers.Dense(
                units=10,
                activation="softmax",
                name="output_probabilities",
            ),
        ],
        name="Handwritten_Digit_ANN",
    )

    # Compile model with Adam optimizer and sparse categorical crossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_architecture_details() -> dict:
    """
    Provides human-readable structural metadata for documentation, reports, and UI.

    Returns:
        Dictionary containing architectural specifications, formulas, and layer roles.
    """
    return {
        "architecture_type": "Fully Connected Artificial Neural Network (Multilayer Perceptron)",
        "input_features": 784,
        "layers": [
            {
                "name": "Input Layer",
                "units": 784,
                "activation": "None (Identity / Pass-through)",
                "parameters": 0,
                "description": "Receives 784 flattened grayscale pixel intensities normalized to [0, 1].",
            },
            {
                "name": "Hidden Layer 1",
                "units": 128,
                "activation": "ReLU (Rectified Linear Unit)",
                "parameters": (784 * 128) + 128,  # 100,480
                "formula": "a_1 = max(0, W_1 * x + b_1)",
                "description": "Extracts non-linear low-to-mid level stroke feature representations.",
            },
            {
                "name": "Hidden Layer 2",
                "units": 64,
                "activation": "ReLU (Rectified Linear Unit)",
                "parameters": (128 * 64) + 64,  # 8,256
                "formula": "a_2 = max(0, W_2 * a_1 + b_2)",
                "description": "Combines feature representations into abstract digit stroke geometries.",
            },
            {
                "name": "Output Layer",
                "units": 10,
                "activation": "Softmax",
                "parameters": (64 * 10) + 10,  # 650
                "formula": "P(y = i | x) = exp(z_i) / sum_j exp(z_j)",
                "description": "Outputs a normalized probability distribution across digits 0 through 9.",
            },
        ],
        "total_trainable_parameters": 100480 + 8256 + 650,  # 109,386
        "loss_function": "Sparse Categorical Cross-Entropy",
        "optimizer": "Adam (Adaptive Moment Estimation)",
        "batch_size": 32,
        "default_epochs": 5,
        "validation_split": 0.10,
    }


if __name__ == "__main__":
    # Demonstration & quick validation of model architecture
    ann = build_ann_model()
    ann.summary()
