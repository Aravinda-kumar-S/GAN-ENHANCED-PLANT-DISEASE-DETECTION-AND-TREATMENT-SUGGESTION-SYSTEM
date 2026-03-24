import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt

# --- GAN CONFIGURATION ---
LATENT_DIM = 100
IMG_SHAPE = (128, 128, 3) # Lightweight GAN
OUTPUT_DIR = "generated_plants"

def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(128 * 16 * 16, input_dim=LATENT_DIM))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((16, 16, 128)))
    
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(3, (3,3), activation='tanh', padding='same'))
    return model

def build_discriminator():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=IMG_SHAPE))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def train_gan(epochs=5000, batch_size=32, save_interval=500):
    # This is a template script. To use this, you must construct a dataset of the specific "Rare Class"
    # and feed it here.
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    gan_input = layers.Input(shape=(LATENT_DIM,))
    img = generator(gan_input)
    discriminator.trainable = False
    validity = discriminator(img)
    gan = models.Model(gan_input, validity)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    
    print(f"🔥 GAN Training Template Ready. Please load your rare class images to 'train_gan.py' to begin generation.")
    # Training loop would go here...

if __name__ == "__main__":
    train_gan()
