import cv2
from random import randrange

# Load some pretrained data on frontal faces from open cv (haar cascade algo)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Choose an image to detect faces in
# img = cv2.imread('rbt.png')
img = cv2.imread('rbt2.jfif')
# img = cv2.imread('grp.jpg')

# Converting the color image into grayscale image
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# Draw rectanles around the face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 5)


# Display the selected image
cv2.imshow('Face Detector', img)

# It keeps the image box open until a key id pressed
cv2.waitKey()





print("Code Ended")