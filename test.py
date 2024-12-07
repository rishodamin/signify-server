import cv2
from model import LevioasModel


# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)
model = LevioasModel()

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Real-time detection loop

while True:

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    corrected_frame = cv2.flip(frame, 1)
    cv2.imshow("Leviosa", corrected_frame)

    # Run YOLO detection on the current frame
    results =   model.predict(frame)# Run detection

   

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()