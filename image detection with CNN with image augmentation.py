import cv2
import cv2 as cv
from tkinter import filedialog
from tkinter import *
from matplotlib.widgets import Button
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

fig = plt.figure(figsize=(5, 5))
fig.canvas.manager.set_window_title('Image Prediction (Statistik)')


global model
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)
train_dataset = train.flow_from_directory('C:/Users/user/OneDrive/Documents/Tugas-Tugas Kuliah Semester 4/Statistik/dataset/training/', target_size=(200,200), batch_size=3, class_mode='binary')
validation_dataset = train.flow_from_directory('C:/Users/user/OneDrive/Documents/Tugas-Tugas Kuliah Semester 4/Statistik/dataset/validation/', target_size=(200,200), batch_size=3, class_mode='binary')

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
model_fit = model.fit(train_dataset, steps_per_epoch=3, epochs=3, validation_data=validation_dataset)

def insertImage(event):
    global img, path
    path = filedialog.askopenfilename()
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axImage1.imshow(img)

def resetImage(event):
    axImage1.clear()

def resetFilter(event):
    global path, img
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axImage1.imshow(img)
    

def rotate90 (event):
    global img
    img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    axImage1.imshow(img)

def rotate180 (event):
    global img
    img=cv2.rotate(img, cv2.ROTATE_180)
    axImage1.imshow(img)

def rotate270 (event):
    global img
    img=cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    axImage1.imshow(img)

def grayscaleFilter (event):
    global img
    if img is not None:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
        axImage1.imshow(img)
    
def redFilter (event):
    global img
    if img is not None:
        img[:,:,1] = 0
        img[:,:,2] = 0
        axImage1.imshow(img)
        
def greenFilter (event):
    global img
    if img is not None:
        img[:,:,0] = 0
        img[:,:,2] = 0
        axImage1.imshow(img)

def blueFilter (event):
    global img
    if img is not None:
        img[:,:,0] = 0
        img[:,:,1] = 0
        axImage1.imshow(img)


def predictImage(event):
    global img
    if img is not None:
        img2 = Image.fromarray(img)
        img2 = img2.resize((200, 200))
        img2 = np.array(img2)
        val = model.predict(np.array([img2]))
        print(val[0][0])
        if val[0][0] < 0.5:
            axImage1.set_title("blueberry")
        else:
            axImage1.set_title("raspberry")


axImage1 = fig.add_axes([1/8, 1/2, 3/4, 5/12])
axPredict = fig.add_axes([0, 1/4, 1, 1/8])
axInsert = fig.add_axes([0, 1/8, 1/2, 1/8])
axReset = fig.add_axes([0, 0, 1/4, 1/8])
axResetFilter = fig.add_axes([1/4, 0, 1/4, 1/8])
axRotate90 = fig.add_axes([3/4, 2/12, 1/4, 1/12])
axRotate180 = fig.add_axes([3/4, 1/12, 1/4, 1/12])
axRotate270 = fig.add_axes([3/4, 0, 1/4, 1/12])
axGrayscale= fig.add_axes([1/2, 1/8,  1/4, 1/8])
axRed= fig.add_axes([1/2, 2/24, 1/4, 1/24])
axGreen= fig.add_axes([1/2, 1/24, 1/4, 1/24])
axBlue= fig.add_axes([1/2, 0, 1/4, 1/24])
# (JARAK DARI KIRI, JARAK DARI BAWAH, LEBAR, PANJANG)

buttonPredict = Button(axPredict, 'predict image')
buttonInsert = Button(axInsert, 'insert image')
buttonReset = Button(axReset, 'reset')
buttonResetFilter = Button(axResetFilter, 'reset filter')
buttonRotate90 = Button(axRotate90, 'rotate 90')
buttonRotate180 = Button(axRotate180, 'rotate 180')
buttonRotate270 = Button(axRotate270, 'rotate -90')
buttonGrayscale = Button(axGrayscale, 'grayscale')
buttonRed = Button(axRed, 'red')
buttonGreen = Button(axGreen, 'green')
buttonBlue = Button(axBlue, 'blue')

buttonPredict.on_clicked(predictImage)
buttonInsert.on_clicked(insertImage)
buttonReset.on_clicked(resetImage)
buttonResetFilter.on_clicked(resetFilter)
buttonRotate90.on_clicked(rotate90)
buttonRotate180.on_clicked(rotate180)
buttonRotate270.on_clicked(rotate270)
buttonGrayscale.on_clicked(grayscaleFilter)
buttonRed.on_clicked(redFilter)
buttonGreen.on_clicked(greenFilter)
buttonBlue.on_clicked(blueFilter)

plt.show()