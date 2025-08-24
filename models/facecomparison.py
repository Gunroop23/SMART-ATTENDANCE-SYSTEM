import numpy as np
import face_recognition
import cv2

image1 = cv2.imread('ImagesBasics/Alakh_pandey.jpeg')
image2 = cv2.imread('ImagesBasics/mukeshAmbani.jpg')

# Check if the images are loaded properly
if image1 is None:
    print("Error loading image 1")
    exit(1)

if image2 is None:
    print("Error loading image 2")
    exit(1)

# Convert the images from BGR (OpenCV format) to RGB (face_recognition format)
img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Print the shape and type of the images for debugging
print("Image 1 shape:", img1.shape, "dtype:", img1.dtype)
print("Image 2 shape:", img2.shape, "dtype:", img2.dtype)
FaceLoc1 = face_recognition.face_locations(img1)[0]
encodeImg1 = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1, (FaceLoc1[3], FaceLoc1[0]), (FaceLoc1[1], FaceLoc1[2]), (255, 0, 255), 2)


faceLoc2 = face_recognition.face_locations(img2)[0]
encodeImg2 = face_recognition.face_encodings(img2)[0]
cv2.rectangle(img2, (faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (255, 0, 255), 2)


results = face_recognition.compare_faces([encodeImg1], encodeImg2)
faceDis = face_recognition.face_distance([encodeImg1], encodeImg2)
print("Comparison results:", results)
print("Face distance:", faceDis)
cv2.putText(img2, f'{results}{round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

# Display images
cv2.imshow('ELON1', img1)
cv2.imshow('ELON2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()