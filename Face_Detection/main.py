# Importing OpenCV Library to perform basic image processing
import cv2
# For the deep learning based Modules and face landmark detection we import Dlib Library
import dlib
# Importing Numpy for related function
import numpy as np
# Using face_utils for basic operation of face detection
from imutils import face_utils

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Marking status for current state
sleep = 0
drowsy = 0
active = 0
yawning = 0
status = ""
color = (0, 0, 0)

# This function is use to calculate euclidean distance between two landmarks
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# This function is use to calculate the distance and ratio of landmarks
def blinked(a, b, c, d, e, f):
    up = compute(b,d) + compute(c, e) # This calculate the distance between two vertical landmarks
    down = compute(a, f) # This calculate the distance between horizantal landmarks
    ratio = up / (2.0 * down) # It's the ratio between up and down

    # Checking if its blinked
    if (ratio > 0.22):
        return 2
    elif (ratio > 0.18 and ratio <= 0.22 ):
        return 1
    else:
        return 0
# This function is use to check the yawn of person
def yawn(g, h):
    opn = compute(g, h)
    return opn

while True:
    _, frame = cap.read() # This function is use to read the frame caputared
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # This OpenCV function is use to convert RGB frame into GraySCale
    fps = cap.get(cv2.CAP_PROP_FPS) # This open CV Function is used for Framerate

# To manage the brightness of frame
    frame_brightness = cv2.convertScaleAbs(frame, 0.25, 2)

# Detecting faces
    faces = detector(gray)
#  Dectecting faces in array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy() # This function is use to copy the frame

    # Creating a rectangle on the face in the frame using co-ordinates
        cv2.rectangle(face_frame, (x1,y1), (x2,y2), (180, 150, 10), 1)

    # Detecting landmarks on the frame
        landmarks = predictor(gray, face)

    # Converting the landmarks into numpy array
        landmarks = face_utils.shape_to_np(landmarks)

    # This number are actually the landmarks which will indicate eye
        left_blink = blinked(landmarks[36], landmarks[37],
                                 landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                                  landmarks[44], landmarks[47], landmarks[46], landmarks[45])
    # This will calculate the distance between the top and bottom of lips of person
        mouth_dist = yawn(landmarks[62], landmarks[66])

        # Condition for Yawning
        if (mouth_dist > 12):
            yawning += 1
            if (yawning > 3):
                status = "Yawning"
                color = (150, 220, 90)

        # To judge what to do if eye is blinking
        if (left_blink == 0 or right_blink == 0):
            sleep += 1
            drowsy = 0
            active = 0
            if (sleep > 3):
                status = "SLEEPING !!!"
                color = (255, 0, 0)

        elif (left_blink == 1 or right_blink == 1):
            sleep = 0
            active = 0
            drowsy += 1

            if (drowsy > 3):
                status = "Drowsy !"
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if (active > 3):
                status = "Active :)"
                color = (0, 255, 0)


        for n in range (0, 68):
            (x, y) = landmarks[n]

            cv2.circle(face_frame, (x, y), 1, (255, 255, 255))
        # To change the frame from BGR to RGB
            image_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        # To create the line between landmarks 62 and 66
            cv2.line(face_frame, tuple(landmarks[62]),tuple(landmarks[66]), color, 2)
        # To create circle on the landmarks
            lip_up_lndm_x, lip_up_lndm_y = landmarks[62]
            cv2.circle(face_frame, (lip_up_lndm_x, lip_up_lndm_y), 3, (255, 255, 255))
            lip_down_lndm_x, lip_down_lndm_y = landmarks[66]
            cv2.circle(face_frame, (lip_down_lndm_x, lip_down_lndm_y), 3, (255, 255, 255))

    fps = int(fps)  # COnversion of float fps to int
    fps = str(fps)  # Conversion of int fps to str

    cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, fps, (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # To Create Sobel Frame
    sobelx = cv2.Sobel(face_frame, cv2.CV_64F, 1 , 0 , ksize = 3)
    sobely = cv2.Sobel(face_frame, cv2.CV_64F, 0 , 1, ksize = 3)

# To create laplacian frame
    laplacian = cv2.Laplacian(frame,cv2.CV_64F)

# To Rotate the image 180degree
    img = cv2.rotate(face_frame, cv2.ROTATE_180)

# For Canny Edge Detection
    t_lower = 50
    t_upper = 150
    Canny_edge = cv2.Canny(frame, t_lower, t_upper)

# For converting in HSV Image
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow('HSV IMAGE', hsvImage) # Output for HSV Image
    cv2.imshow("180 degree", img) # Output for Rotate Image
    cv2.imshow('Canny_edge',Canny_edge) # Output for Canny Edge
    cv2.imshow('sobelx',sobelx) # Output for Sobel x
    cv2.imshow('sobely',sobely) # Output for Sobel y
    cv2.imshow('laplacian',laplacian) # Output for Laplacian
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL) # Output for window resize
    cv2.resizeWindow("Frame", 500, 800) # Output for resizing the window
    cv2.imshow("Frame", image_rgb) # Output for image conversion
    cv2.imshow("Result of detector", frame) # Output
    cv2.imshow("Brightness", frame_brightness) # Output for Brightness


    key = cv2.waitKey(1)
    if key == 27:
        break
