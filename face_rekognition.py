import cv2

# Loaded some pre-trained data faces frontal profile
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Chosed image to detect
img = cv2.imread('easy.png')
# Grayscaling
gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)

# sprint(face_coordinates)

# Draw rectangle
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+y, w+h), (0, 0, 255), 2)

cv2.imshow('Faces_detected', gray_scaled_img)
cv2.waitKey()





print("Code Complete!")