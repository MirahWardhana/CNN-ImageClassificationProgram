import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tkinter import filedialog
from tkinter import *
from matplotlib.widgets import Button
from PIL import Image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

fig = plt.figure(figsize=(5, 5))
fig.canvas.manager.set_window_title('Image Prediction (Statistik)')

global model
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

#change the directory path into your own training and validation dataset path where you save the folder
train_dataset = train.flow_from_directory('C:/Users/user/OneDrive/Documents/Tugas-Tugas Kuliah Semester 4/Statistik/tugas statistik/dataset/training/', target_size=(200,200), batch_size=3, class_mode='binary')
validation_dataset = train.flow_from_directory('C:/Users/user/OneDrive/Documents/Tugas-Tugas Kuliah Semester 4/Statistik/tugas statistik/dataset/validation/', target_size=(200,200), batch_size=3, class_mode='binary')

train_dataset.class_indices
train_dataset.classes

a=train_dataset.classes
print(a)
b=train_dataset.class_indices
print(b)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200,200,3)),
    tf.keras.layers.MaxPool2D (2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D (2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPool2D (2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
model_fit = model.fit(train_dataset, steps_per_epoch=3, epochs=10, validation_data=validation_dataset)

def insertImage(event):
    global img, path
    path = filedialog.askopenfilename()
    img = cv2.imread(path)
    print(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    axImage1.imshow(img)


def reset(event):
    axImage1.clear()


def predictImage(event):
    img = Image.open(path).convert('RGB')
    img = img.resize((200, 200))
    img = np.array(img) / 255.0 
    val = model.predict(np.array([np.array(img)]))
    if val[0][0] < 0.5:
        axImage1.set_title("blueberry")
    else:
        axImage1.set_title("raspberry")


axImage1 = fig.add_axes([1/8, 1/2, 3/4, 5/12])
axPredict = fig.add_axes([0, 1/4, 1, 1/8])
axInsert = fig.add_axes([0, 0, 1/2, 1/4])
axReset = fig.add_axes([1/2, 0, 1/2, 1/4])

buttonPredict = Button(axPredict, 'predict image')
buttonInsert = Button(axInsert, 'insert image')
buttonReset = Button(axReset, 'reset')

buttonPredict.on_clicked(predictImage)
buttonInsert.on_clicked(insertImage)
buttonReset.on_clicked(reset)

plt.show()