import tensorflow as tf
import numpy as np
import os
import cv2

generator = tf.keras.models.load_model('C:/Program Files/Python39/AVATAR GENRATION/gan_model.h5')  

data_dir = r"C:\Program Files\Python39\AVATAR GENRATION\lfw"

output_dir = 'C:\Program Files\Python39\AVATAR GENRATION\generated_avatars'
os.makedirs(output_dir, exist_ok=True)


num_samples = 15  
latent_dim = 100  

for i in range(num_samples):
    noise = np.random.normal(0, 1, (1, latent_dim))  
    generated_image = generator.predict(noise)[0]    
    generated_image = (generated_image + 1) / 2.0    

    
    saved_path = os.path.join(output_dir, f'generated_avatar_{i}.png')
    cv2.imwrite(saved_path, (generated_image * 255).astype(np.uint8))

print("Avatars generated and saved successfully!")
