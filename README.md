# 🐦 Bird Audio Classification — Project Overview

This project builds a deep learning system for **multi-label bird species classification** from raw environmental audio. The core idea is to transform raw waveforms into **mel spectrograms**, which provide a structured time–frequency representation, and then use a **convolutional neural network (ResNet-18)** to learn discriminative acoustic features.

---

##  Problem Setup

Given an audio signal:
  
x[n] ∈ ℝ  

we aim to predict a vector of probabilities:

y ∈ [0,1]^C  

where C is the number of bird species and each entry represents the probability that a species is present in the clip.

This is a **multi-label classification problem** (not mutually exclusive classes).

---

## 🎧 Signal Processing Pipeline

### 1. Short-Time Fourier Transform (STFT)

We first convert the 1D time-domain signal into a time-frequency representation:

X(t, ω) = Σ x[n] · w[n − t] · e^(−iωn)

Where:
- w[n] is a window function  
- t is time  
- ω is frequency  

---

### 2. Power Spectrogram

We compute the energy at each time-frequency bin:

|X(t, ω)|^2

---

### 3. Mel Frequency Mapping

Human perception of sound is nonlinear, so we map frequencies to the mel scale:

m(f) = 2595 · log10(1 + f / 700)

---

### 4. Mel Filter Bank Projection

We apply triangular filters over the frequency axis:

S_mel(t, m) = Σ |X(t, ω)|^2 · H_m(ω)

Where:
- H_m(ω) is the m-th mel filter  

---

### 5. Log Scaling

Final input to the model:

S_log(t, m) = log(S_mel(t, m) + ε)

This stabilizes variance and improves training.

---

##  Model

The model is a modified ResNet-18:

f_θ : ℝ^(T × M) → ℝ^C  

Where:
- T = time frames  
- M = mel bins  
- C = number of classes  

Output logits:

z = f_θ(x)

---

## 🎯 Prediction Layer

Apply sigmoid activation:

ŷ = σ(z)

σ(z) = 1 / (1 + e^(−z))

Each output is an independent probability for each class.

---

##  Loss Function

Binary Cross Entropy (multi-label):

L = − Σ [ y log(ŷ) + (1 − y) log(1 − ŷ) ]

Equivalently with logits:

L = − Σ [ y log(σ(z)) + (1 − y) log(1 − σ(z)) ]

---

##  Optimization

Parameters θ are optimized via gradient descent:

θ ← θ − η ∇_θ L

Using:
- AdamW optimizer  
- Learning rate scheduling  

---

##  Key Intuition

- Audio → spectrogram converts raw signal into structured features  
- Mel scaling emphasizes perceptually relevant frequencies  
- CNN learns spatial patterns (time × frequency)  
- Multi-label sigmoid allows overlapping bird calls  

---

##  Summary

This pipeline combines:

Signal Processing:
- STFT  
- Mel filter banks  
- Log scaling  

Machine Learning:
- Convolutional neural networks  
- Multi-label classification  
- Probabilistic outputs via sigmoid  

to solve a challenging real-world problem: **identifying multiple bird species in noisy soundscapes**.
