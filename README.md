<<<<<<< HEAD
# Handwritten Digit Recognition Using Artificial Neural Network (ANN)

An educational, beginner-friendly, and complete implementation of an Artificial Neural Network (Multilayer Perceptron) for handwritten digit recognition (0–9) using Python, TensorFlow/Keras, and Streamlit.

**Developed by:** **Sarkar Tuhassinul Arnob**  
*ICT Student and AI/ML Project Developer*  
**Department of ICT, Islamic University, Bangladesh**  
- **Email:** [arnob.ict.iu@gmail.com](mailto:arnob.ict.iu@gmail.com)  
- **WhatsApp:** [+8801811860339](https://wa.me/8801811860339)  
- **GitHub:** [https://github.com/TuhassinulArnob](https://github.com/TuhassinulArnob)  
- **LinkedIn:** [https://www.linkedin.com/in/tuhassinularnob/](https://www.linkedin.com/in/tuhassinularnob/)  

---

## 1. Project Title
**Handwritten Digit Recognition Using a Fully Connected Artificial Neural Network (ANN)**

---

## 2. Abstract
Handwritten digit recognition is a canonical benchmark problem in computer vision and machine learning. This project presents a complete, modular, and beginner-accessible implementation of a Multilayer Perceptron (MLP) trained on the MNIST database. Rather than relying on Convolutional Neural Networks (CNNs), this system deliberately implements a dense feedforward Artificial Neural Network (784 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 10) to clearly illustrate foundational principles: matrix multiplication, non-linear activation functions (ReLU, Softmax), gradient backpropagation, and loss optimization with Adam. A specialized image preprocessing pipeline bridges the domain gap between freehand user drawings and MNIST formatting standards. An interactive Streamlit web application allows users to draw digits in real time, view intermediate preprocessing stages, and observe prediction confidences and probability distributions.

---

## 3. Introduction
Optical Character Recognition (OCR) and handwritten digit classification form the cornerstone of modern automated document processing, postal sorting, and bank cheque verification. In academic settings, handwritten digit recognition serves as an ideal bridge between introductory machine learning theory and deep neural network architectures. This project demonstrates how a classical fully connected neural network, implemented with clean Python code and runnable on standard consumer CPU hardware, achieves $>97.5\%$ accuracy on official benchmark tests while providing real-time interactive inference.

---

## 4. Problem Statement
Given an arbitrary grayscale image representing a single handwritten numerical digit between 0 and 9, classify the image into its true numerical category $\{0, 1, 2, 3, 4, 5, 6, 7, 8, 9\}$ with high accuracy and quantify the network's predictive certainty. Furthermore, provide a robust bridging mechanism that allows a model trained on standardized historical data to accurately process arbitrary freehand digital drawings produced by mice or touchscreens.

---

## 5. Objectives
1. **Algorithmic Understanding:** Implement a fully connected feedforward Artificial Neural Network (MLP) without convolutional layers, clarifying the core mechanics of dense layers, activations, and parameter optimization.
2. **Reproducible Training:** Train the ANN on the official MNIST training split using deterministic seeds, mini-batch gradient descent (Adam), and sparse categorical cross-entropy loss.
3. **Rigorous Evaluation:** Quantitatively evaluate the network on the 10,000-sample test partition, computing accuracy, loss, precision, recall, F1-score, and a confusion matrix.
4. **Domain Adaptation Pipeline:** Construct a preprocessing pipeline that normalizes canvas drawings (polarity, aspect-ratio scaling, center-of-mass alignment) to match MNIST formatting.
5. **Interactive Demonstration:** Deliver a responsive, modern Streamlit graphical interface suitable for undergraduate classroom presentations.

---

## 6. Dataset Description
The model is trained and benchmarked on the **MNIST (Modified National Institute of Standards and Technology)** dataset:

- **Source:** Automatically fetched and cached via `tf.keras.datasets.mnist.load_data()`.
- **Training Set:** 60,000 grayscale images ($28 \times 28$ pixels) with integer labels $0 \dots 9$.
- **Validation Split:** 10% (6,000 samples) reserved from training data for hyperparameter validation.
- **Testing Set:** 10,000 independent test images with ground-truth labels.
- **Input Dimensions:** Each image is $28 \times 28$ pixels. For the ANN, each 2D image matrix is flattened into a 1D vector of $28 \times 28 = 784$ continuous features.
- **Pixel Intensity Normalization:** Raw uint8 values ranging from $0$ to $255$ are scaled linearly to the interval $[0.0, 1.0]$:
  $$\hat{x} = \frac{x}{255.0}$$
- **Data Integrity:** The official 10,000 test images are never exposed to the model during training or validation, ensuring zero data leakage.

---

## 7. ANN Architecture
The neural network is a 4-layer fully connected architecture (Input, two Hidden layers, and Output):

```
[Input Layer: 784 Features]
            │
            ▼  (Weights: 784 x 128 = 100,352 | Biases: 128)
[Hidden Layer 1: 128 Neurons, ReLU Activation]
            │
            ▼  (Weights: 128 x 64 = 8,192 | Biases: 64)
[Hidden Layer 2: 64 Neurons, ReLU Activation]
            │
            ▼  (Weights: 64 x 10 = 640 | Biases: 10)
[Output Layer: 10 Neurons, Softmax Activation]
```

### Layer Breakdown & Parameter Counts:
| Layer | Layer Type | Output Shape | Activation | Trainable Parameters | Mathematical Formula |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | Feature Vector | $(784,)$ | Identity | $0$ | $\mathbf{x} \in \mathbb{R}^{784}$ |
| **Dense 1** | Fully Connected | $(128,)$ | ReLU | $784 \times 128 + 128 = 100,480$ | $\mathbf{a}_1 = \max(0, \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$ |
| **Dense 2** | Fully Connected | $(64,)$ | ReLU | $128 \times 64 + 64 = 8,256$ | $\mathbf{a}_2 = \max(0, \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2)$ |
| **Output** | Fully Connected | $(10,)$ | Softmax | $64 \times 10 + 10 = 650$ | $p_i = \frac{\exp(z_{3,i})}{\sum_j \exp(z_{3,j})}$ |
| **Total** | | | | **109,386 Parameters** | |

---

## 8. Theory of Artificial Neural Networks
An Artificial Neural Network is a collection of interconnected processing nodes (neurons) structured into sequential layers. Unlike classical rule-based algorithms, an ANN learns an empirical mapping function $f: \mathbb{R}^{784} \rightarrow \mathbb{R}^{10}$ directly from training observations by iteratively adjusting internal weight matrices $\mathbf{W}$ and bias vectors $\mathbf{b}$.

### The Artificial Neuron
An individual neuron performs a linear combination of its input vector followed by a non-linear mapping:
$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$
$$a = f(z)$$
where:
- $\mathbf{x} \in \mathbb{R}^d$ is the vector of incoming inputs.
- $\mathbf{w} \in \mathbb{R}^d$ represents the synaptic weights.
- $b \in \mathbb{R}$ is the neuron's bias scalar.
- $z$ is the pre-activation net input.
- $f(\cdot)$ is an activation function providing non-linear representational capacity.

---

## 9. Forward Propagation
During forward propagation, signals flow deterministically from the input layer to the output layer:

1. **Hidden Layer 1:**
   $$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 \quad \text{where } \mathbf{W}_1 \in \mathbb{R}^{128 \times 784}, \; \mathbf{b}_1 \in \mathbb{R}^{128}$$
   $$\mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1)$$

2. **Hidden Layer 2:**
   $$\mathbf{z}_2 = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2 \quad \text{where } \mathbf{W}_2 \in \mathbb{R}^{64 \times 128}, \; \mathbf{b}_2 \in \mathbb{R}^{64}$$
   $$\mathbf{a}_2 = \text{ReLU}(\mathbf{z}_2)$$

3. **Output Layer:**
   $$\mathbf{z}_3 = \mathbf{W}_3 \mathbf{a}_2 + \mathbf{b}_3 \quad \text{where } \mathbf{W}_3 \in \mathbb{R}^{10 \times 64}, \; \mathbf{b}_3 \in \mathbb{R}^{10}$$
   $$\hat{\mathbf{y}} = \text{Softmax}(\mathbf{z}_3)$$

---

## 10. Activation Functions

### A. Rectified Linear Unit (ReLU)
The hidden layers utilize the ReLU activation function:
$$\text{ReLU}(z) = \max(0, z)$$
- **Characteristics:**
  - Derivative is $1$ for $z > 0$ and $0$ for $z < 0$.
  - Avoids the vanishing gradient problem encountered with Sigmoidal and Hyperbolic Tangent ($\tanh$) functions.
  - Highly computationally efficient to evaluate on CPU hardware.

### B. Softmax Activation
The final layer applies the Softmax activation function to convert raw logits $\mathbf{z}_3$ into a normalized probability distribution across the 10 digit classes:
$$P(y = i \mid \mathbf{x}) = \frac{\exp(z_i)}{\sum_{j=0}^{9} \exp(z_j)}$$
- **Characteristics:**
  - Guarantees that $0 \le P(y = i) \le 1$ for all $i \in \{0, \dots, 9\}$.
  - Enforces the axiom of probability: $\sum_{i=0}^{9} P(y = i) = 1$.
  - Exponentiation accentuates differences between the largest logit and competing classes.

---

## 11. Loss Function
The model measures error using **Cross-Entropy Loss**. For a single sample with a one-hot ground-truth label vector $\mathbf{y}$, Categorical Cross-Entropy is defined as:
$$\mathcal{L} = -\sum_{i=0}^{9} y_i \log(p_i)$$

### Sparse Categorical Cross-Entropy in Implementation:
Because our dataset labels are stored as natural integers $y \in \{0, 1, \dots, 9\}$ rather than 10-dimensional one-hot vectors, we use **Sparse Categorical Cross-Entropy**:
$$\mathcal{L} = -\log(p_{y_{\text{true}}})$$
- Mathematically equivalent to categorical cross-entropy.
- Saves significant memory by avoiding creation of 60,000 ten-dimensional one-hot arrays.
- Heavily penalizes confident incorrect predictions (as $p_{\text{true}} \rightarrow 0, \mathcal{L} \rightarrow \infty$).

---

## 12. Backpropagation and Weight Updates
Training optimizes the network's weights $\mathbf{W}$ and biases $\mathbf{b}$ to minimize the total loss $\mathcal{L}$ over the training dataset.

1. **Gradient Computation via the Chain Rule:**
   Backpropagation computes the partial derivative of the scalar loss with respect to each parameter:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_l} \cdot \left(\mathbf{a}_{l-1}\right)^T$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_l}$$
   where the error vector $\boldsymbol{\delta}_l = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_l}$ is propagated backwards layer-by-layer:
   $$\boldsymbol{\delta}_{l} = \left(\mathbf{W}_{l+1}^T \boldsymbol{\delta}_{l+1}\right) \odot f'(\mathbf{z}_l)$$

2. **Parameter Update:**
   Trainable parameters are adjusted in the opposite direction of the gradient to descend the loss surface.

---

## 13. Optimizer: Adam (Adaptive Moment Estimation)
Rather than standard Stochastic Gradient Descent (SGD) with a fixed learning rate, the network utilizes the **Adam optimizer**:
- **First Moment Vector (Moving Average of Gradients):**
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
- **Second Moment Vector (Moving Average of Squared Gradients):**
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
- **Bias Correction:**
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
- **Parameter Update:**
  $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
Default parameters: $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$.

---

## 14. Training and Testing Procedure
1. **Dataset Partitioning:**
   - 60,000 total training samples.
   - 10% (6,000 samples) allocated to internal validation monitoring.
   - 54,000 samples used for gradient calculation.
   - 10,000 samples reserved exclusively for unbiased final test evaluation.
2. **Mini-Batch Execution:**
   - Mini-batch size: $32$ samples.
   - Number of training batches per epoch: $54,000 / 32 = 1,688$ batches.
3. **Epochs:**
   - $5$ complete passes over the training dataset.
4. **Reproducibility:**
   - NumPy seed: `42`.
   - TensorFlow seed: `42`.

---

## 15. Image Preprocessing Pipeline
Freehand canvas drawings inherently differ from the clean, centered MNIST dataset. Without normalization, feedforward ANNs struggle because dense layers lack translation invariance.

Our preprocessing pipeline (`preprocessing.py`) rigorously conforms canvas inputs to MNIST conventions:

```
[Raw User Canvas (RGBA)]
         │
         ▼
[1. Empty Canvas Verification] ──(Blank)──> Raise EmptyCanvasError
         │ (Valid content)
         ▼
[2. Polarity Normalization] ──> White stroke on Black background [0, 255]
         │
         ▼
[3. Noise Filtering] ──> Suppress faint stray pixels (< 20 threshold)
         │
         ▼
[4. Bounding Box Cropping] ──> Crop tightly around digit strokes
         │
         ▼
[5. Aspect-Ratio Resizing] ──> Scale digit to fit ~20x20 box (Lanczos)
         │
         ▼
[6. Center-of-Mass Alignment] ──> Place inside 28x28 frame and shift by (dy, dx)
         │
         ▼
[7. Pixel Normalization] ──> Divide by 255.0 to yield [0.0, 1.0] range
         │
         ▼
[Outputs: (28, 28) Preview Array & (1, 784) Model Input Vector]
```

---

## 16. Results and Evaluation Metrics
*(Measured directly on the official 10,000-sample MNIST test set upon executing `python train.py` and `python evaluate.py`)*

### Measured Summary Metrics (Actual Model Execution):
- **Official Test Set Size:** 10,000 independent images
- **Test Accuracy:** **97.40%**
- **Test Loss (Sparse Categorical Cross-Entropy):** **0.0937**
- **Macro Precision:** **97.41%**
- **Macro Recall:** **97.37%**
- **Macro F1-Score:** **97.38%**
- **Weighted Precision:** **97.44%**
- **Weighted Recall:** **97.40%**
- **Weighted F1-Score:** **97.40%**

#### Per-Class Breakdown (Official MNIST Test Set):
| Digit Class | Precision (%) | Recall (%) | F1-Score (%) | Support (Count) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 98.87% | 97.86% | 98.36% | 980 |
| **1** | 99.29% | 99.12% | 99.21% | 1,135 |
| **2** | 98.80% | 95.74% | 97.24% | 1,032 |
| **3** | 94.05% | 98.51% | 96.23% | 1,010 |
| **4** | 97.54% | 96.95% | 97.24% | 982 |
| **5** | 98.40% | 96.52% | 97.45% | 892 |
| **6** | 95.00% | 99.16% | 97.04% | 958 |
| **7** | 97.11% | 98.15% | 97.63% | 1,028 |
| **8** | 97.50% | 96.00% | 96.74% | 974 |
| **9** | 97.58% | 95.74% | 96.65% | 1,009 |

### Visual Artifacts Generated in `outputs/`:
- `confusion_matrix.png`: Heatmap demonstrating accurate classification across all 10 digits with minimal off-diagonal confusion.
- `training_curves.png`: Training vs validation loss and accuracy curves illustrating stable convergence without severe overfitting.
- `sample_predictions.png`: Visual grid of test samples showing ground truth vs predicted labels and model confidence percentages.
- `evaluation_metrics.json`: Complete numerical record of metrics.

---

## 17. Installation Instructions (Windows PowerShell)

Follow these exact steps in **Windows PowerShell**:

### Step 1: Clone or Navigate to Project Directory
```powershell
cd c:\Users\USER\Desktop\Projects\HandWritingANN
```

### Step 2: Create a Python Virtual Environment
*(Note: Python 3.10–3.12 is recommended for TensorFlow compatibility)*
```powershell
py -3.12 -m venv .venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once)*

### Step 4: Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 18. Usage Instructions

### Step 1: Run Preprocessing Unit Tests
Ensure the preprocessing pipeline functions correctly:
```powershell
pytest -v
```

### Step 2: Train the Neural Network
Train the ANN on the MNIST dataset (caches MNIST locally, evaluates on test data, and saves `models/mnist_ann_model.keras`):
```powershell
python train.py
```
*To force retraining from scratch:*
```powershell
python train.py --force
```

### Step 3: Run Model Evaluation & Generate Plots
Compute test metrics, confusion matrix, and sample grids:
```powershell
python evaluate.py
```

### Step 4: Launch the Streamlit Web Application
Start the interactive digit drawing and recognition interface:
```powershell
streamlit run app.py
```
Open your web browser at the URL shown (typically `http://localhost:8501`).

---

## 19. Project Structure

```
HandWritingANN/
├── app.py                      # Interactive Streamlit UI and prediction workflow
├── config.py                   # Centralized developer branding & application constants
├── train.py                    # Dataset loading, model training, and artifact saving
├── model.py                    # Fully connected ANN architecture definition
├── preprocessing.py            # Canvas image preprocessing pipeline (MNIST standard)
├── evaluate.py                 # Test-set evaluation, confusion matrix, and reports
├── requirements.txt            # Pinned, compatible Python library dependencies
├── README.md                   # Comprehensive project documentation
├── pytest.ini                  # Pytest runner configuration
├── .gitignore                  # Git tracking rules excluding virtual environments & caches
├── models/
│   ├── .gitkeep                # Tracks models folder
│   └── mnist_ann_model.keras   # Saved Keras model (109,386 parameters)
├── outputs/
│   ├── .gitkeep                # Tracks outputs folder
│   ├── confusion_matrix.png    # Evaluation confusion matrix plot
│   ├── training_curves.png     # Loss and accuracy curves across epochs
│   ├── sample_predictions.png  # Sample test predictions grid
│   ├── training_history.json   # Recorded epoch loss/accuracy logs
│   └── evaluation_metrics.json # Full test metrics dictionary
└── tests/
    └── test_preprocessing.py   # Automated pytest unit tests for preprocessing
```

---

## 20. Limitations
1. **Lack of Translation Invariance:** Because fully connected layers assign specific weights to specific flattened pixel coordinates, an ANN is sensitive to extreme shifts or off-center digits if not properly centered by the preprocessing pipeline.
2. **Single-Digit Constraint:** The network expects a single isolated digit ($0 \dots 9$). Multi-digit strings or non-numeric characters cannot be recognized without a localization/segmentation step.
3. **Canvas vs Pen-and-Paper Domain Shift:** Touchpad and mouse strokes have different stroke velocities, widths, and edge anti-aliasing than scanned pen-on-paper ink.

---

## 21. Future Improvements
1. **Data Augmentation:** Introduce slight random rotations ($\pm 10^\circ$), zooms, and shears during training to increase robustness to handwriting variance.
2. **Convolutional Neural Networks (CNN):** Extend to a CNN architecture (e.g., LeNet-5 or modern Conv2D-MaxPooling networks) to compare translation invariance with the ANN baseline.
3. **Multi-Digit OCR:** Add Connected Component Analysis (CCA) or Contour Detection to segment and classify multi-digit strings (e.g., zip codes, phone numbers).
4. **Model Quantization:** Convert the model to TensorFlow Lite (`.tflite`) or ONNX for ultra-lightweight edge deployment.

---

## 22. References
1. **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).** *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278-2324.
2. **Kingma, D. P., & Ba, J. (2014).** *Adam: A method for stochastic optimization.* arXiv preprint arXiv:1412.6980.
3. **Nair, V., & Hinton, G. E. (2010).** *Rectified linear units improve restricted boltzmann machines.* In Proceedings of the 27th International Conference on Machine Learning (ICML-10).
4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press.

---

## 23. Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `No matching distribution found for tensorflow` | Python version is $>3.12$ (e.g., Python 3.14). | Create venv using Python 3.12: `py -3.12 -m venv .venv`. |
| `Script execution is disabled on this system` | Windows PowerShell execution policy restriction. | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. |
| `Model not found at models/mnist_ann_model.keras` | Web app opened prior to model training. | Run `python train.py` in the terminal or use the button in the UI. |
| `EmptyCanvasError: No digit detected` | The canvas was blank or the stroke was too faint. | Draw a clear digit in the center of the canvas and click Predict. |
| Digit is misclassified | Drawing stroke is too thin or drawn far off-center. | Use the sidebar slider to set stroke thickness to 18px and draw near the center. |

---

## 24. Academic Presentation Demonstration Script

When presenting this project to an instructor or academic committee, follow this 3-minute demonstration script:

1. **Introduction (30 seconds):**
   > *"Good morning. Today I am presenting our Handwritten Digit Recognition system developed using an Artificial Neural Network in Python. Rather than using a complex Convolutional Neural Network, our objective is to demonstrate the fundamental mechanics of a Multilayer Perceptron: 784 input features, two hidden layers with ReLU activations, and a 10-class Softmax probability distribution."*

2. **Training & Architecture Explanation (45 seconds):**
   > *"The network trains on 60,000 MNIST images using Adam optimization and Sparse Categorical Cross-Entropy. Over 5 epochs, the model optimizes 109,386 trainable parameters, achieving an official test accuracy of 97.40% on 10,000 previously unseen test images."*

3. **Evaluation Metrics Walkthrough (45 seconds):**
   > *"On our Model Evaluation tab, we can observe the confusion matrix. Notice that the diagonal elements show high sample counts with minimal misclassification, such as between 4 and 9 or 3 and 8. The training and validation curves verify stable convergence without overfitting."*

4. **Live Interactive Recognition (60 seconds):**
   > *"Finally, here is the live Streamlit interface. When I draw a digit '8' on the canvas, our preprocessing pipeline crops the bounding box, resizes it to 20x20 pixels preserving the aspect ratio, centers it by center-of-mass, and normalizes it to match the MNIST format. As you can see, the ANN predicts '8' with high confidence, and the probability distribution displays the certainty across all 10 classes."*

---

## 25. Deployment Guide: GitHub Publishing & Streamlit Community Cloud

This project is structured for 1-click cloud deployment on **Streamlit Community Cloud** (free hosting provided by Streamlit/Snowflake).

### Step A: Publish Project to GitHub

1. Initialize Git in the project root:
   ```bash
   git init
   ```
2. Stage and commit all project files:
   ```bash
   git add .
   git commit -m "feat: complete handwritten digit recognition ANN web platform"
   ```
3. Set the default branch and connect your remote repository:
   ```bash
   git branch -M main
   git remote add origin https://github.com/TuhassinulArnob/HandWriting_Detection_simpleANN.git
   ```
4. Push your codebase:
   ```bash
   git push -u origin main
   ```

### Step B: Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Fill in the repository configuration:
   - **Repository:** `TuhassinulArnob/HandWriting_Detection_simpleANN`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Under **Advanced settings**, set:
   - **Python Version:** `3.12`
5. Click **"Deploy!"**.
   Streamlit Cloud will automatically clone the repository, install dependencies from `requirements.txt`, load the pre-trained `.keras` model, and provide a public URL (e.g., `https://handwritingann-arnob.streamlit.app/`).

---

## 26. Future Vercel Deployment & Migration Architecture

### Can Streamlit be deployed directly to Vercel?
**No.** Streamlit cannot run directly as a standard Vercel frontend because:
1. **Stateful WebSocket Architecture:** Streamlit requires a persistent, long-running Python process maintaining bidirectional WebSockets to track execution state.
2. **Serverless Execution Constraints:** Vercel specializes in static frontend assets and stateless Serverless/Edge Functions that terminate after each HTTP request ($<15$s timeouts).
3. **Package Size Quotas:** Full TensorFlow distributions exceed Vercel's 50MB uncompressed serverless function deployment limits.

### Recommended Full-Stack Architecture for Vercel:

If a future version is deployed on Vercel, adopt a decoupled architecture:

```
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│     Vercel Frontend (Next.js)   │   HTTP    │   Inference API (FastAPI)       │
│  - React / Tailwind CSS         │ ────────> │  - Hosted on Render / Railway   │
│  - HTML5 Canvas Drawing Board   │  POST     │  - ONNX Runtime / TF Lite       │
│  - Client-side pre-scaling      │  /predict │  - Returns JSON: {digit, conf}  │
└─────────────────────────────────┘           └─────────────────────────────────┘
```

#### Migration Phases:
1. **Model Export (ONNX / TF Lite):**
   Convert `mnist_ann_model.keras` to ONNX or TensorFlow Lite (`.tflite`). This shrinks the runtime requirement from 400MB TensorFlow down to lightweight `<20MB` `onnxruntime`, allowing sub-10ms inference.
2. **Backend API (FastAPI / Flask):**
   Wrap `preprocessing.py` and the ONNX model into a 40-line FastAPI service:
   ```python
   @app.post("/predict")
   async def predict_digit(file: UploadFile):
       ...
       return {"digit": int(pred), "confidence": float(conf), "probabilities": probs}
   ```
   Deploy the backend to **Render**, **Railway**, or **Fly.io** (with free Python/Docker hosting).
3. **Frontend on Vercel:**
   Create a Next.js 14 / React application with `react-sketch-canvas`, submitting the drawn image via `fetch('/api/predict')`. Deploy instantly via `vercel deploy`.

*Note: The existing Streamlit application remains the complete, self-contained, working baseline.*
