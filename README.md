# AVATAR GENERATION USING GAN
GAN-based deep learning system for generating avatar images from real images.
A system capable of generating personalized avatar images from real photographs of individuals. These avatars should capture the essence of the person while stylizing them in a unique and artistic way.

## Dataset
Real Face Image Dataset

![Link](https://drive.google.com/drive/folders/0B7EVK8r0v71pQ3NzdzRhVUhSams?resourcekey=0-Kpdd6Vctf-AdJYfS55VULA&usp=drive_link)

## Requirements

- Python3.8
- tensorflow
- numpy
- opencv
- matplotlib

## Steps to run
#### Installing Dependencies
```
pip install -r requirements.txt
```

#### Training
For training and building of GAN model
```
python3 train_model.py
```

#### Generating avatar
To generate avatar
```
python3 generator_model.py
```

### Evaluation
#### GAN LOSS CURVES
#### INCEPTION SCORE
![Inception](images\inception/_score.png)
