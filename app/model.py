import tensorflow as tf
import cv2
import numpy as np

def detect_emotion_in_image(image):
    # Read the image from the provided file
    frame = cv2.imread(image)

    # Load the pre-trained neural network model
    final_model = tf.keras.models.load_model('app/static/model_cv.h5')

    # Create a CascadeClassifier to detect faces
    faceCascade = cv2.CascadeClassifier("app/static/haarcascade_frontalface_default.xml")

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Check if any faces are detected
    if len(faces) == 0:
        return "No Face Detected"  # Return a message if no face is detected

    # Iterate through each detected face
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect facial features within the face (e.g., eyes)
        faces = faceCascade.detectMultiScale(roi_gray)

        if len(faces) == 0:
            return "Face not Detected"  # Return a message if facial features are not detected
        else:
            for (ex, ey, ew, eh) in faces:
                face_roi = roi_color[ey:ey+eh, ex:ex+ew]

    # Resize the face image to fit the model
    final_image = cv2.resize(face_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image/255.0

    # Make predictions for the emotion
    predictions = final_model.predict(final_image)
    prediction = np.argmax(predictions)

    # Create a dictionary to map prediction numbers to emotion labels
    emotion_dict = {
        0: "Sedih",
        1: "Netral",
        2: "Bahagia",
        3: "Marah"
    }

    # Return the detected emotion
    return emotion_dict[prediction]

# # testing
# image = "app/static/Capture.PNG"
# detected_emotion = detect_emotion_in_image(image)
# print("Emotion: " + detected_emotion)