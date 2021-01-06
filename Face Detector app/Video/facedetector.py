import cv2
from random import randrange

# Load some pretrained data on frontal faces from open cv (haar cascade algo)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture the webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    
    # Read current frame
    successful_frame_read, frame = webcam.read()

    # Converting the color frmae into grayscale frame
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_frame)
    print(face_coordinates)

    # Draw rectanles around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 5)

    # Display the selected image
    cv2.imshow('Face Detector', frame)

    # It keeps the image box open until a key id pressed
    key = cv2.waitKey(1)

    # Quit the app
    if key==81 or key==113:
        break

# Release the webcam 
webcam.release()

print("Code Ended")