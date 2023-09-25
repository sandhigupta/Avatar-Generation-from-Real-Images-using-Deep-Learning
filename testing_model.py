import tensorflow as tf
import numpy as np
import os
from scipy.stats import entropy
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image

# Load generated images from the output directory
output_dir = 'C:\Program Files\Python39\AVATAR GENRATION\generated_avatars'
generated_images = []

for filename in os.listdir(output_dir):
    if filename.endswith(".png"):
        img = image.load_img(os.path.join(output_dir, filename), target_size=(299, 299))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        generated_images.append(img)

generated_images = np.array(generated_images)

# Load a pre-trained InceptionV3 model
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

def calculate_inception_score(images, inception_model, splits=10):
    scores = []
    n_images = len(images)

    for i in range(splits):
        start_idx = i * (n_images // splits)
        end_idx = (i + 1) * (n_images // splits)

        subset = images[start_idx:end_idx]

        # Predict class probabilities for the subset using InceptionV3
        preds = inception_model.predict(subset)

        p_yx = np.mean(preds, axis=0)

        p_y = np.expand_dims(np.mean(p_yx, axis=0), axis=0)

        kl_divergence = entropy(p_yx.T, p_y.T)
        score = np.exp(np.mean(kl_divergence))

        scores.append(score)

    return np.mean(scores), np.std(scores)

mean_score, std_score = calculate_inception_score(generated_images, inception_model)
print(f'Inception Score: {mean_score} (±{std_score})')