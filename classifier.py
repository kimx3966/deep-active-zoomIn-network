

from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping
from keras.models import model_from_json

import random
import os

import numpy as np
import cv2

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt




zoom1_background = os.listdir('./data/gt_img/zoom1_background')
print("zoom1_background:", len(zoom1_background))
zoom2_background = os.listdir('./data/gt_img/zoom2_background')
print("zoom2_background:", len(zoom2_background))
zoom3_background = os.listdir('./data/gt_img/zoom3_background')
print("zoom3_background:", len(zoom3_background))
zoom4_background = os.listdir('./data/gt_img/zoom4_background')
print("zoom3_background:", len(zoom4_background))


output_path = "./data/label_false/"

sample_z1 = random.sample(zoom1_background, 250)
cnt=0
input_path = './data/gt_img/zoom1_background/'
for img_name in sample_z1:
    img = cv2.imread(input_path+img_name,1)
    cv2.imwrite(output_path+'background_%d.png' % cnt,img)
    cnt+=1
    
sample_z2 = random.sample(zoom2_background, 250)
input_path = './data/gt_img/zoom2_background/'
for img_name in sample_z2:
    img = cv2.imread(input_path+img_name,1)
    cv2.imwrite(output_path+'background_%d.png' % cnt,img)
    cnt+=1 
    
    
sample_z3 = random.sample(zoom3_background, 250)
input_path = './data/gt_img/zoom3_background/'
for img_name in sample_z3:
    img = cv2.imread(input_path+img_name,1)
    cv2.imwrite(output_path+'background_%d.png' % cnt,img)
    cnt+=1 
    
    
sample_z4 = random.sample(zoom4_background, 250)
input_path = './data/gt_img/zoom4_background/'
for img_name in sample_z4:
    img = cv2.imread(input_path+img_name,1)
    cv2.imwrite(output_path+'background_%d.png' % cnt,img)
    cnt+=1 





input_path_true = "./data/label_true/"
input_path_false = "./data/label_false/"


train_imgs = np.zeros((2000,120,120,3), dtype=np.int)
for i in range(0,1000):
    img = cv2.imread(input_path_true+"z1_%d.png" % i,1)
    train_imgs[i,:,:,:] = img

    
for i in range(1000,2000):
    img = cv2.imread(input_path_false+"background_%d.png" % (i-1000),1)
    train_imgs[i,:,:,:] = img


train_labels = np.zeros((2000,), dtype=np.int)
for i in range(0,1000):
    train_labels[i] = 1

for i in range(1000,2000):
    train_labels[i] = 0

    
print("train_imgs:",train_imgs.shape)
print("train_labels:", train_labels.shape)





num_train = 1200
num_val = 400
num_test = 400
idxs = list(range(0,2000))

train_idxs = random.sample(idxs, num_train)
X_train = train_imgs[train_idxs]
y_train = train_labels[train_idxs]
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)



rest_idxs = []
for i in idxs:
    if not (i in train_idxs):
        rest_idxs.append(i)

val_idxs = random.sample(rest_idxs, num_val)
X_val = train_imgs[val_idxs]
y_val = train_labels[val_idxs]
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)



test_idxs = []
for i in rest_idxs:
    if not (i in val_idxs):
        test_idxs.append(i)

X_test = train_imgs[test_idxs]
y_test = train_labels[test_idxs]
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# In[6]:


train_idxs = list(range(0,600))+ list(range(1000,1600))
X_train = train_imgs[train_idxs]
y_train = train_labels[train_idxs]
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

val_idxs = list(range(600,800))+ list(range(1600,1800))
X_val = train_imgs[val_idxs]
y_val = train_labels[val_idxs]
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

test_idxs = list(range(800,1000))+ list(range(1800,2000))
X_test = train_imgs[test_idxs]
y_test = train_labels[test_idxs]
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# In[7]:


#Load the VGG model
image_size = 120 
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
plot_model(vgg16_model, to_file='pretrained_vgg16.png')


# In[8]:


# Freeze the layers except the last 4 layers
# for layer in vgg16_model.layers:
#     layer.trainable = True
    
for layer in vgg16_model.layers:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in vgg16_model.layers:
    print(layer, layer.trainable)


# In[9]:


# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg16_model)
 
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
 

# # Split training, validation, and testing data


# Compile the model
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience=1, restore_best_weights=True)


# Train the model
history = model.fit(X_train,y_train,validation_data=(X_val,y_val),callbacks=[early_stopping_monitor], epochs=5, verbose=1)



y_raw_pred = model.predict(X_test)
y_pred = np.round(y_raw_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: " + str(accuracy))



# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")




# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 



