<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=013243&height=220&section=header&text=Pure%20NumPy%20CNN&fontSize=60&fontAlignY=38&desc=Conquering%20MNIST%20&%20EMNIST%20from%20Scratch&descAlignY=60&descAlign=50&fontColor=ffffff" />

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
    <img src="https://img.shields.io/badge/Numba-000?style=for-the-badge&logo=numba&logoColor=white">
    <img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge">
  </p>
</div>

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Benchmarks](#-benchmarks)
- [Quick Start](#-quick-start)
- [Architecture & Math](#-architecture--math)
- [Visualizations & Feature Maps](#-visualizations--feature-maps)

---

## 🚀 Overview
This repository features a **from-scratch Convolutional Neural Network (CNN)** implementation. By avoiding high-level frameworks like PyTorch or TensorFlow, I engineered a custom backpropagation engine to master the fundamental calculus and matrix operations governing Deep Learning.

---

## 📊 Benchmarks
The model was rigorously tested on standard handwritten digits and complex alphanumeric sets.

| Dataset | Classes | Training Acc | Testing Acc |
| :--- | :---: | :---: | :---: |
| **MNIST** | 10 | 99.98% | **98.88%** |
| **EMNIST** | 47 | 92.16% | **86.04%** |

---

<div>
  <h2>🧠 Architecture & MathThe</h2>
  architecture is purposefully lean, designed to maximize generalization through a single-pass convolutional feature extractor:
  
  Conv Layer: 16 Learned Kernels (3x3), ReLU Activation, Max Pooling (2x2).
  Dense ANN: Flatten -> 128 Hidden -> 64 Hidden -> Output Layer.
  
  <h2>Advanced Optimizations</h2>
    To achieve production-grade performance using pure NumPy, I implemented several advanced computational strategies:
    <p>⚡ Numba JIT Compilation: Wrapped core matrix operations in @njit(fastmath=True) to execute at C-level speeds, overcoming the Python interpreter bottleneck.</p>
    <p>🧬 Synthetic Data Augmentation: Expanded the feature space by applying affine transformations via scipy to 25k samples.</p>
    <p>🛡️ L2 Regularization & Decay: Stabilized gradient descent and closed a 10% overfitting gap by manually deriving and applying L2 penalties to the weight updates:</p>
               <p>$$W_{new} = W_{old} - \alpha \left( \frac{\partial L}{\partial W} + \lambda W_{old} \right)$$</p>
      <p>(Where $\alpha$ smoothly decays at a rate of 0.85 per epoch)</p> 
    <h2>🔍 Visualizations & Feature Maps</h2>
    To truly understand what the CNN is "learning," we have to look inside the hidden layers.
    <h3>How the Kernels Work</h3>
    In this architecture, I implemented 16 custom kernels (3x3 matrices). During the forward pass, these kernels convolve (slide) across the 28x28 input image. Each kernel acts as a specialized filter, trained strictly via backpropagation to detect specific low-level features such as horizontal lines, vertical edges, or sharp loops.
    <h3>Inside the Brain</h3>
    When an image is passed through these 16 kernels and the ReLU activation function (which drops negative values to introduce non-linearity), it produces 16 distinct "Feature Maps."Below is a visualization of what the CNN actually "sees" inside its first layer when looking at a handwritten digit. (Notice how bright yellow areas indicate high activation—this is where a specific kernel successfully found the geometric shape it was trained to look for!)
</div>

## 💻 Quick Start

To run this model locally and see the training in action:

```bash
# 1. Clone the repository
git clone [https://github.com/Jit338/E-MNIST.git](https://github.com/Jit338/E-MNIST.git)

# 2. Navigate into the directory
cd E-MNIST

# 3. Install required mathematical packages
pip install numpy numba scipy scikit-learn matplotlib

# 4. Launch the Jupyter Notebook
jupyter notebook CNN2.ipynb
