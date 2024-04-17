#real-time emotion recognition using a webcam feed.

#Import libraries
import cv2
from keras.models import model_from_json
import numpy as np

#maps integer labels to their corresponding emotional expressions.
emotion_label = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}

#loads the model architecture from a JSON file and loads the pre-trained weights into the model.
json_file=open("C:\\Users\\pro\\PycharmProjects\\open cv\\best_emotion_model.json","r")
loaded_model_json=json_file.read()
json_file.close()
emotion_model=model_from_json(loaded_model_json)
emotion_model.load_weights("C:\\Users\\pro\\PycharmProjects\\open cv\\best_emotion_model.weights.h5")
print("loaded model")

# Initialize the camera
capture = cv2.VideoCapture(0)

#real-time emotion recognition using a webcam feed.
while True:                             #initiates an infinite loop to continuously capture frames from the webcam.
    ret, frame = capture.read()
    if not ret:
        break

    face_cascade = cv2.CascadeClassifier("C:\\Users\\pro\\Desktop\\techolas\\deep learning\\haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_rect = face_cascade.detectMultiScale(gray, 1.1, 9)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 2)
        gray_frame = gray[y:y + w, x:x + w] #Extracts the region of interest (ROI) containing the face from the grayscale frame
        cropped_image = np.expand_dims(np.expand_dims(cv2.resize(gray_frame, (48, 48)), -1), 0)
        emotion_pred = emotion_model.predict(cropped_image)
        maxindex = int(np.argmax(emotion_pred))
        cv2.putText(frame, emotion_label[maxindex], (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Detected faces', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()




