wget = 'https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.goodhousekeeping.com%2Flife%2Fpets%2Fg4531%2Fcutest-dog-breeds%2F&psig=AOvVaw1PeJTS8mKvW95FkXPCQL4x&ust=1757161203529000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCIjX9_TNwY8DFQAAAAAdAAAAABAJ'

import matplotlib.image as npimg
import matplotlib.pyplot as plt
import numpy as np

# converting the img as a numpy array
img = npimg.imread('dog.jpg')
# print(img.shape)


# displaying the img from numpy array
img_plot = plt.imshow(img)
# plt.show()


# resizing the image using Pillow library
from PIL import Image
img = Image.open('dog.jpg')
resized_img = img.resize((200, 150))

# displaying the resized img
resized_img.save('dog_image_resized.png')
res_img = plt.imshow(resized_img)
# plt.show()

# converting img to grayscale 
import cv2
img_gray = cv2.imread('dog_image_resized.png')
grayscale_img = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
print(grayscale_img)