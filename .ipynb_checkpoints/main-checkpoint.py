import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='assets/sign_ml.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

results = []
image = mp.Image.create_from_file('2.jpg')

recognition_result = recognizer.recognize(image)

top_gesture = recognition_result.gestures[0][0]
hand_landmarks = recognition_result.hand_landmarks
results.append((top_gesture, hand_landmarks))
print(results[0])

