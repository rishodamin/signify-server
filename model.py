import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision





class LevioasModel:
    def __init__(self) -> None:
        self.base_options = python.BaseOptions(model_asset_path='assets/sign_ml.task')
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def predict(self, numpy_image):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        recognition_result = self.recognizer.recognize(image)
        if not (recognition_result.gestures):
            return ''
        top_gesture = recognition_result.gestures[0][0]
        return top_gesture.category_name
        

# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# class LevioasModel:
#     def __init__(self) -> None:
#         # Set up base options for gesture recognizer
#         self.base_options = python.BaseOptions(model_asset_path='assets/combined_gesture_model.task')
#         self.options = vision.GestureRecognizerOptions(base_options=self.base_options)
#         self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

#     def predict(self, numpy_image):
#         # Convert numpy image to mediapipe image format
#         image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

#         # Recognize gestures from the image
#         recognition_result = self.recognizer.recognize(image)

#         # Check if there are any detected gestures
#         if not recognition_result.multi_hand_landmarks:
#             return ''
        
#         # Check if both hands are detected
#         if len(recognition_result.multi_hand_landmarks) == 2:
#             # Extract landmarks for both hands
#             left_hand_gesture = recognition_result.gestures[0][0]  # First detected hand
#             right_hand_gesture = recognition_result.gestures[1][0]  # Second detected hand

#             # Combine the gestures (you could concatenate or map them to a combined gesture)
#             combined_gesture = self.get_combined_gesture(left_hand_gesture.category_name, right_hand_gesture.category_name)

#             return combined_gesture

#         # If only one hand is detected, return that hand's gesture
#         elif len(recognition_result.multi_hand_landmarks) == 1:
#             top_gesture = recognition_result.gestures[0][0]
#             return top_gesture.category_name
        
#         return ''

#     def get_combined_gesture(self, left_gesture, right_gesture):
#         """
#         A method to combine the gestures from both hands and return a label.
#         This is where you define your logic for combining gestures.
#         For example, you can create mappings for specific combinations.
#         """
#         # Simple example of combined gesture logic
#         if left_gesture == "fist" and right_gesture == "open_hand":
#             return "right_open_left_fist"
#         elif left_gesture == "open_hand" and right_gesture == "fist":
#             return "left_open_right_fist"
#         # Add more combined gesture logic as needed
#         return f"{left_gesture}_{right_gesture}"
