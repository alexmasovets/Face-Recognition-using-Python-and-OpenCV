import cv2
import sqlite3

path = 'haarcascade_frontalcatface.xml' # Path to cascade
camera_number = 0 # Camera number
object_name = 'FACE' # Object name to display
frameWidth = 640 # Display width
frameHeight = 480 # Display height
color = (255, 0, 0)

# Connect to SQLite database
conn = sqlite3.connect('detected_objects.db')
c = conn.cursor()

# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS objects
             (x INTEGER, y INTEGER, w INTEGER, h INTEGER)''')

cap = cv2.VideoCapture(camera_number)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

# Create trackbar

cv2.namedWindow("Result")
cv2.resizeWindow("Result", frameWidth, frameHeight + 100)
cv2.createTrackbar("Scale", "Result", 400, 1000, empty)
cv2.createTrackbar("Neig", "Result", 8, 20, empty)
cv2.createTrackbar("Min Area", "Result", 0, 100000, empty)
cv2.createTrackbar("Brightness", "Result", 180, 255, empty)
cv2.createTrackbar("Night Vision", "Result", 0, 255, empty)

# Load the classifiers downloaded

cascade = cv2.CascadeClassifier(path)

while True:
# Set camera brightness from trackbar value
    camera_brightness = cv2.getTrackbarPos("Brightness", "Result")
    cap.set(10, camera_brightness)
# Set camera night vision from trackbar value
    camera_night_vision = cv2.getTrackbarPos("Night Vision", "Result")
    cap.set(10, camera_night_vision)
# Get camera image and convert to grayscale
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect the object using the cascade
    scale_value = 1 + (cv2.getTrackbarPos("Scale", "Result") / 1000)
    neig = cv2.getTrackbarPos("Neig", "Result")
    objects = cascade.detectMultiScale(gray, scale_value, neig)
# Display the detected objects
    for(x, y, w, h) in objects:
        area = w * h
        min_area = cv2.getTrackbarPos("Min Area", "Result")
        if area < min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, object_name, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            roi_color = img[y:y + h, x:x + w]
            # Store the detected object's information in the SQLite database
            c.execute("INSERT INTO objects (x, y, w, h) VALUES (?, ?, ?, ?)", (x, y, w, h))

    conn.commit()
    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break