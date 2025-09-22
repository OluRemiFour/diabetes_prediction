# Extract Dataset using Kaggle API

# extracting the compressed dataset
from zipfile import ZipFile
dataset = 'dogs-vs-cats.zip'

with ZipFile(dataset, 'r') as zip:
    zip.extractall()
    print('The dataset is extracted')
    
    
# extract the trained dataset
from zipfile import ZipFile
dataset = 'path-to-trained-file'

with ZipFile(dataset, 'r') as zip:
    zip.extractall()
    print('Trained Dataset is extracted') 

# counting the number of files on train folder
import os
path, dirs, files = next(os.walk('train'))
file_count = len(files)
print("Number of images:", file_count )
    
# Printing the names of images
file_names = os.listdir('train')
print(file_names)

# import dependencies
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as npimg
from sklearn.model_selection import train_test_split

# displaying the image for dog
img = npimg.imread('train/name-of-dog.jpg')
imgplt = plt.imshow(img)
plt.show()

# displaying the image for cat
img = npimg.imread('train/name-of-cat.jpg')
imgplt = plt.imshow(img)
plt.show()

# iterate and count the number of dogs and cats img
file_names = os.listdir('train')

dog_count = 0
cat_count = 0

for img_name in file_names:
    name = img_name[0:3]
    
    if name == 'dog':
        dog_count += 1
    else:
        cat_count += 1

print('Number of dog images = ', dog_count)
print('Number of cat images = ', cat_count)


# Resizing all the images
# creating a dir for resized images
os.mkdir('image_resized')

original_folder = '/train/'
resized_folder = '/image_resized/'

for i in range(2000):
    filename = os.listdir(original_folder)[i]
    img_path = original_folder+filename
    
    img = Image.open(img_path)
    # resize
    # convert to RGB

    #  . . .more
    

# loop through the resized image and label the cats as 0 and dog as 1
#  create an empty array named labels
#  loop; for i in range(2000), i.e run in 2000 times
#  for each file_names[i], get the first 3 letter; [0:3] and save in a variable
#  if saved variable == dog: labels.append(1), else labels.append(1)


# counting the images of dogs and cats out of 2000 images
# values, counts = np.unique(labels, return_counts=True)
# print(counts)
# print(values)