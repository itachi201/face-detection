import cv2

# Loaded some pre-trained data faces frontal profile
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# Chosed image to detect
img = cv2.imread('easy.png')
# Grayscaling
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
face_coordinates = trained_face_data.detectMultiScale(gray, 1.3, 4)

# Draw rectangle
for (x, y, w, h) in face_coordinates:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
   # eyes = eye_cascade.detectMultiScale(roi_gray)
   # for (ex, ey, ew, eh) in eyes:
     #   cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('Faces_detected', img)
cv2.waitKey()





print("Code Complete!")