# Mini-Project--Application-of-NN

## Project Title:
KITCHEN UTENSIL CLASSIFIER
## Project Description 
A kitchen utensil is a small hand held tool used for food preparation. Common kitchen tasks include cutting food items to size, heating food on an open fire or on a stove, baking, grinding, mixing, blending, and measuring etc.., In this project we will classify the utensils using Classification techniques of neural networks by inputting a image 

## Algorithm:
* Import the packages.
* Read the images.
* Using classes and epoch find the accuracy and array.
* Using array we find the name of the name of the utensil.
## Program:
```
/*
Program to implement 
Developed by   : Srinivasan S
RegisterNumber : 212220230048  
*/
```
```python
import splitfolders  # or import split_folders
splitfolders.ratio("Raw", output="output", seed=1337, ratio=(.9, .1), group_prefix=None) # default values
import matplotlib.pyplot as plt
import matplotlib.image as mping
img = mping.imread("output/val/BREAD_KNIFE/breadkniferaw2.jpg")

plt.imshow(img)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)
train = train_datagen.flow_from_directory("output/train/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")
test = train_datagen.flow_from_directory("output/val/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")

from tensorflow.keras.preprocessing import image
test_image = image.load_img('output/val/BREAD_KNIFE/breadkniferaw2.jpg', target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = tf.expand_dims(test_image,axis=0)
test_image = test_image/255.
test_image.shape
import tensorflow_hub as hub
m = tf.keras.Sequential([
hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"),
tf.keras.layers.Dense(20, activation='softmax')
])
m.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])
history = m.fit(train,epochs=5,steps_per_epoch=len(train),validation_data=test,validation_steps=len(test))
classes=train.class_indices
classes=list(classes.keys())
m.predict(test_image)
classes[tf.argmax(m.predict(test_image),axis=1).numpy()[0]]
import pandas as pd
pd.DataFrame(history.history).plot()
m.summary()
```
## Output:
![image](https://user-images.githubusercontent.com/103049243/205567898-8dabfc49-3974-46f5-8079-6725c3f51357.png)
![image](https://user-images.githubusercontent.com/103049243/205567964-a695a523-5ab7-4832-bed6-c9839e04f7b0.png)
![image](https://user-images.githubusercontent.com/103049243/205568025-a7908c2f-334e-4ce7-95e9-acf67e6fd197.png)

## Advantage :
* We can identify the utensils with this program.
* It is easy to use.
* The reaction speed is fast.
## Result:
Thus a kitchen utensil classifier is created using Neural Networks


## Project By:
### Srinivasan S - 212220230048
