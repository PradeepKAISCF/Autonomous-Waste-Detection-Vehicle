import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import serial
import time


# Load the trained model
model = load_model("C:/Users/Admin/Desktop/waste detection model/Res.keras")

# Define class names (assuming binary classification: animal vs no animal)
class_names =  ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Function to preprocess the frames

def preprocess_frame(frame):
    # Convert frame to PIL Image and ensure it's in RGB format
    image = Image.fromarray(frame)
    image = image.convert('RGB')

    # Resize the image to match the input size of the model
    image = image.resize((256, 256))

    # Convert the image to a NumPy array and normalize the pixel values
    image_array = np.array(image) / 255.0

    # Add batch dimension (1, 256, 256, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# Start the video capture (0 for the first camera, you can change if using an external camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Make a prediction
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions)
    acc = np.max(predictions)
    print(acc)
    if acc >=0.90:
        print(acc)
        print(class_names[predicted_class])

    # Get the predicted label
    #label = class_names[predicted_class]

    # Display the label on the frame
        cv2.putText(frame, class_names[predicted_class], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the label
    cv2.imshow('Real-Time waste Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
