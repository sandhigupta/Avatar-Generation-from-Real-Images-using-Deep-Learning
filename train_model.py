import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, Activation
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

latent_dim = 100
dataset_dir = r'C:\Program Files\Python39\AVATAR GENRATION\lfw'

# Generator model
def build_generator(latent_dim):
    generator = models.Sequential()
    generator.add(layers.Dense(128 * 8 * 8, input_dim=latent_dim))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Reshape((8, 8, 128)))
    generator.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))  
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv2D(3, (3, 3), padding='same'))
    generator.add(layers.Activation('tanh'))
    return generator

# Discriminator model
def build_discriminator(input_shape):
    discriminator = models.Sequential()
    discriminator.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    discriminator.add(layers.BatchNormalization(momentum=0.8))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    discriminator.add(layers.BatchNormalization(momentum=0.8))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    discriminator.add(layers.BatchNormalization(momentum=0.8))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dropout(0.4))
    discriminator.add(layers.Dense(1, activation='sigmoid'))
    return discriminator

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    model = models.Model(gan_input, gan_output)
    return model

input_shape = (64, 64, 3)

# Build and compile the model
discriminator = build_discriminator(input_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
generator = build_generator(latent_dim)
generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
discriminator.trainable = False
model = build_gan(generator, discriminator)
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# Load and preprocess your dataset here
dataset_dir = r'C:\Program Files\Python39\AVATAR GENRATION\lfw'
image_paths = os.listdir(dataset_dir)
print(image_paths)

# Define the input shape for resizing
input_shape = (64, 64)

# Initialize an empty list to store preprocessed images
preprocessed_images = []

# Walk through the subdirectories and collect image files
for root, dirs, files in os.walk(dataset_dir):
    for filename in files:
        # Check if the file has a valid image extension (e.g., .jpg, .png, .jpeg)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # Construct the full path to the image file
            image_path = os.path.join(root, filename)

            # Read and preprocess the image
            img = cv2.imread(image_path)
            if img is not None:
                # Resize the image to the desired input shape
                img = cv2.resize(img, (input_shape[0], input_shape[1]))

                # Normalize the image to [-1, 1]
                img = (img.astype(np.float32) - 127.5) / 127.5

                # Append the preprocessed image to the list
                preprocessed_images.append(img)

# Convert real_images to a NumPy array
preprocessed_images = np.array(preprocessed_images)

epochs = 5
batch_size = 64
sample_interval = 1000

# Initialize lists for storing losses
d_losses = []
g_losses = []

# Training loop
for epoch in range(epochs):
    # Training the discriminator
    idx = np.random.randint(0, len(preprocessed_images), batch_size)
    real_batch = preprocessed_images[idx]
    real_labels = np.ones((batch_size, 1))
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_batch = generator.predict(noise)
    fake_labels = np.zeros((batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch(real_batch, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_batch, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gan_labels = np.ones((batch_size, 1))
    g_loss = model.train_on_batch(noise, gan_labels)

    d_losses.append(d_loss[0])
    g_losses.append(g_loss)
    
    # Save the generator's output
    if epoch % sample_interval == 0:
        print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

        noise_batch = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise)
        
        
        
        for i, generated_image in enumerate(generated_images):
            save_path = f'generated_image_epoch_{epoch}_sample_{i}.jpg'
            generated_image = (generated_image + 1) / 2.0  
            cv2.imwrite(save_path, (generated_image * 255).astype(np.uint8))

# Plot the loss curves
plt.figure(figsize=(16, 8))
plt.plot(range(epochs), d_losses, label="Discriminator Loss")
plt.plot(range(epochs), g_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Loss Curves")
plt.show()

# Generate images
num_samples = 10
noise_batch = np.random.normal(0, 1, (num_samples, latent_dim))
generated_images = generator.predict(noise_batch)

# Save the generator model
generator.save("generator_model.h5")

# Save the discriminator model
discriminator.save("discriminator_model.h5")

# Save the GAN model
model.save("gan_model.h5")
