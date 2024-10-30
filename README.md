# Deep Learning Neural Networks

This repository provides an overview and implementations of various types of neural networks used in deep learning, covering foundational concepts and common architectures. Each network type has unique strengths and is suited for different applications, ranging from image and text analysis to complex, multidimensional data modeling.

## Table of Contents
1. [Introduction](#introduction)
2. [Types of Neural Networks](#types-of-neural-networks)
    - [1. Feedforward Neural Network (FNN)](#1-feedforward-neural-network-fnn)
    - [2. Convolutional Neural Network (CNN)](#2-convolutional-neural-network-cnn)
    - [3. Recurrent Neural Network (RNN)](#3-recurrent-neural-network-rnn)
    - [4. Long Short-Term Memory Network (LSTM)](#4-long-short-term-memory-network-lstm)
    - [5. Autoencoder](#5-autoencoder)
    - [6. Generative Adversarial Network (GAN)](#6-generative-adversarial-network-gan)
    - [7. Transformer](#7-transformer)
3. [Installation](#installation)
4. [Contributions](#contributions)


## Introduction

Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to model complex patterns in data. This repository contains example implementations of several popular neural network architectures. Each network has unique features and applications, allowing them to address diverse data types and tasks.

## Types of Neural Networks

### 1. Feedforward Neural Network (FNN)
Feedforward Neural Networks are the simplest type of neural network architecture where connections do not form cycles. Information moves only in one direction—from input to output. They are typically used for basic classification and regression tasks.

#### Example Code
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 2. Convolutional Neural Network (CNN)
Convolutional Neural Networks are designed to process data with grid-like topology, such as images. They use convolutional layers to extract spatial features from the input. CNNs are widely used for image recognition, object detection, and other computer vision tasks.

#### Example Code
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 3. Recurrent Neural Network (RNN)
Recurrent Neural Networks are designed for sequential data, where the output from one step is fed as input to the next step. They have "memory" that captures information from previous inputs. RNNs are commonly used in time series prediction, natural language processing, and speech recognition.

#### Example Code
```python
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(50, activation='relu', input_shape=(time_steps, features)),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 4. Long Short-Term Memory Network (LSTM)
LSTM networks are a type of RNN designed to address the vanishing gradient problem. They have gates (input, forget, output) that control the flow of information, making them capable of learning long-term dependencies. LSTMs are often used in text generation, machine translation, and other sequential tasks.

#### Example Code
```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(time_steps, features)),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 5. Autoencoder
Autoencoders are neural networks used for unsupervised learning of representations. They consist of an encoder that reduces the input into a compressed representation and a decoder that reconstructs the original input. Autoencoders are widely used in dimensionality reduction, anomaly detection, and noise removal.

#### Example Code
```python
input_img = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_img)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
```

### 6. Generative Adversarial Network (GAN)
GANs are a type of neural network used for generating new data samples similar to a given dataset. They consist of two networks: a generator and a discriminator that work in a minimax game. GANs are popular for generating realistic images, videos, and even text.

#### Example Code
```python
from tensorflow.keras.layers import Dense, Reshape, Flatten, Input
from tensorflow.keras.models import Model

# Generator
generator_input = Input(shape=(latent_dim,))
x = Dense(128, activation='relu')(generator_input)
x = Dense(784, activation='sigmoid')(x)
generator = Model(generator_input, x)

# Discriminator
discriminator_input = Input(shape=(784,))
x = Dense(128, activation='relu')(discriminator_input)
x = Dense(1, activation='sigmoid')(x)
discriminator = Model(discriminator_input, x)
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# GAN model
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
fake_img = generator(gan_input)
gan_output = discriminator(fake_img)
gan = Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')
```

### 7. Transformer
Transformers are attention-based networks that handle sequential data without relying on recurrence. They can process data in parallel, making them highly efficient for tasks like language modeling, translation, and text generation. Transformers are the foundation of models like BERT and GPT.

#### Example Code
```python
from transformers import TFAutoModel, AutoTokenizer

model = TFAutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("Sample text", return_tensors="tf")
outputs = model(inputs)
```

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/deep-learning-networks.git
cd deep-learning-networks
pip install -r requirements.txt
```

## Contributions

Contributions are welcome! Please open an issue or submit a pull request to contribute.

---

This README provides an overview of different neural network types, with example code for each, making it easy for users to understand and experiment with various architectures in deep learning.
