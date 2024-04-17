#importing libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import datasets,layers,models
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# load image data from directories and generate batches of augmented image data for training and testing
train_data=ImageDataGenerator(rescale=1./255,rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,horizontal_flip=True,fill_mode='nearest').flow_from_directory(
   'C:\\Users\\pro\\PycharmProjects\\open cv\\emotion\\train',
   target_size=(48,48),
   batch_size=64,
   color_mode='grayscale',
   class_mode='categorical'
)
test_data=ImageDataGenerator(rescale=1/255).flow_from_directory(
    'C:\\Users\\pro\\PycharmProjects\\open cv\\emotion\\test',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)


# Calculate class weights
class_weights = {}
total_samples = train_data.samples #Total number of samples in the training dataset.
num_classes = len(train_data.class_indices) #Number of classes in the dataset.

# Calculate the total number of samples for each class
class_samples = np.sum(train_data.labels, axis=0)

print("Number of classes:", num_classes)
print("Class samples:", class_samples)

for i in range(num_classes):
    class_weight = total_samples / (num_classes * class_samples + 1) if class_samples != 0 else 1
    class_weights[i] = class_weight



#Convolutional neural network (CNN) model for emotion recognition
emotion_model = Sequential() #Initializes a sequential model where layers are added sequentially.

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.1)) #helps prevent overfitting
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.1))
emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.1))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.1))
emotion_model.add(Dense(7, activation='softmax')) #Adds the output layer with 7 neurons (one for each emotion class) and softmax activation,
                                                       # which outputs probabilities of each class.



emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])




# Define the filepath where the best model will be saved.Initializes a callback function for saving the best model weights during training.
model_checkpoint = ModelCheckpoint('best_emotion_model.weights.keras',
                                   monitor='val_accuracy',
                                   mode='max',
                                   save_best_only=True,
                                   verbose=1)


# Train the model with the ModelCheckpoint callback
model_info = emotion_model.fit(
    train_data,
    steps_per_epoch=28709 // 64,          #The number of steps (batches of samples) to yield from the generator per epoch.
    epochs=50,                            #Number of epochs
    validation_data=test_data,
    validation_steps=7178 // 64,          #The number of steps (batches of samples) to yield from the validation generator per epoch
    class_weight=class_weights,           #specifies the class weights to be used during training.
    callbacks=[model_checkpoint]
)


#save model
model_json=emotion_model.to_json()
with open("best_emotion_model.json","w") as json_file:
    json_file.write(model_json)

emotion_model.save_weights('best_emotion_model.weights.h5')