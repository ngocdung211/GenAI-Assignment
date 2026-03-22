# Fashion MNIST Image Synthesis using GAN & DCGAN

## Introduction
This project is Homework 1 for the CS646 Generative AI and Applications course at Yuan Ze University. The objective is to implement, train, and compare the performance of two Generative Adversarial Networks (GANs) in synthesizing clothing images based on the Fashion MNIST dataset.

**Authors:** Lam Nguyen, Dung Do, Elise Pierre

## Dataset
*   **Dataset:** Fashion MNIST.
*   **Volume:** Comprises 60,000 training examples.
*   **Dimensions:** Each sample is a $28\times28$ pixel grayscale image.
*   **Preprocessing:** Pixels are normalized to the interval [-1, 1] to match the Generator's output range.

## Model Architecture
The project compares two main architectures:
1.  **Baseline GAN (v1):** Built on Fully Connected (MLP) layers with LeakyReLU activation and He Uniform weight initialization. Upsampling is performed via Dense Mapping.
2.  **Improved DCGAN (v2):** A Deep Convolutional architecture that replaces the MLP with a spatial upsampling mechanism using `Conv2DTranspose` layers. The model features refined Batch Normalization with a momentum of 0.9 and $\epsilon=1e-5$, along with Glorot Uniform initialization to maintain consistent variance and prevent exploding gradients.

## Evaluation Results
*   **Visual Fidelity:** The DCGAN effectively mitigates the "salt-and-pepper" noise and blurred blobs observed in the baseline model, synthesizing recognizable items with clearer boundaries and improved spatial coherence.
*   **Training Stability:** The loss curves of the improved model demonstrate much smoother convergence, reaching a stable adversarial equilibrium compared to the high-frequency oscillations of the baseline MLP.

## Setup Instructions (Google Colab)
This project is configured to run on Jupyter Notebook / Google Colab.

**1. Install required packages:**
```bash
pip install numpy==1.19.2 tensorflow==2.4 torch==1.7.0 torchvision==0.8.1 matplotlib ipython
