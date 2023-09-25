import os
import cv2
import numpy as np
from tqdm import tqdm

data_dir = r"C:\Program Files\Python39\AVATAR GENRATION\lfw"  
output_dir = 'C:\Program Files\Python39\AVATAR GENRATION\generated_avatars'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for filename in tqdm(os.listdir(data_dir)):
    if filename.endswith('.jpg'):
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)
        
    
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
       
        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            
            face_output = os.path.join(output_dir, f'{filename.split(".")[0]}_face_{i}.jpg')
            cv2.imwrite(face_output, face)

print("Face detection completed.")

