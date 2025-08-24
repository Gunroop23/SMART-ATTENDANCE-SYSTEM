import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

def load_images_and_names(path):
    images = []
    classNames = []
    try:
        myList = os.listdir(path)
        print("Image files:", myList)

        for cl in myList:
            img_path = os.path.join(path, cl)
            try:
                curImg = cv2.imread(img_path)
                if curImg is None:
                    print(f"Failed to load image {img_path}")
                    continue

                # Convert image to RGB format if not already in RGB
                if curImg.shape[2] == 3:
                    curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)

                images.append(curImg)
                classNames.append(os.path.splitext(cl)[0])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        print("Class names:", classNames)

    except Exception as e:
        print(f"Error accessing directory {path}: {e}")

    return images, classNames


def findEncodings(images):
    encodeList = []
    for idx, img in enumerate(images):
        try:
            # Print image details for debugging
            print(f"Processing image {idx + 1}/{len(images)}: dtype={img.dtype}, shape={img.shape}")

            # Ensure image is RGB and 8-bit
            if img.dtype != np.uint8 or img.shape[2] != 3:
                print(f"Skipping image {idx + 1} with unsupported type: {img.dtype}, shape: {img.shape}")
                continue

            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                encode = face_encodings[0]
                encodeList.append(encode)
            else:
                print(f"No face found in image {idx + 1}.")
        except Exception as e:
            print(f"Error processing image {idx + 1}: {e}")

    return encodeList

def markAttendance(name):
    try:
        with open('Attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = [line.split(',')[0] for line in myDataList]
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
    except Exception as e:
        print(f"Error marking attendance: {e}")

# Path to the images
path = 'ImagesAttendance'

# Load images and class names
images, classNames = load_images_and_names(path)

# Call the function to find encodings
try:
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
except Exception as e:
    print(f"Error during encoding: {e}")

cap =cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS ,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0.255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
