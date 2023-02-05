import cv2
import numpy as np
import math
import winsound

# Load a pre-trained Haar cascade classifier for object detection
# face_cascade = cv2.CascadeClassifier('cars.xml')
face_cascade = cv2.CascadeClassifier('myhaar.xml')


KNOWN_DISTANCE = 45 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES

video_src = 'video1.avi'

# Load a video capture object
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('real2.mp4')


def calculate_distance(x1, y1, x2, y2):
    # Calculating distance

    # return (((x2 - x1)** 2 + (y2 - y1)**2) ** 0.5)
    # Calculating distance
    return ((x2-x1) * 50) / y2-y1


    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) * 1.0)

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected object
    for (x,y,w,h) in faces:
        # Calculate the distance from the car to the object
        distance = calculate_distance(x, y, w, h)

        # Draw a rectangle around the object
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # If the distance is less than 2 meters, display an alert
        if distance < 2:
            cv2.putText(frame, "ALERT: Object too close!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            winsound.Beep(500, 60)

    # Display the processed frame
    cv2.imshow('frame',frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()


# distance finder function
# def distance_finder(real_object_width, width_in_frmae):
#     distance = (real_object_width * 50) / width_in_frmae
#     return distance