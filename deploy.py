# Deploy your trained model(.h5) to the target machine

print('Script is working...')

import tensorflow as tf
import cv2
import numpy as np


# ----------Custom Inputs for Scripts----------
target_size = (640, 640)
# ----------That's ALL You Need to Do----------



# Load kreas model named xx.h5
model = tf.keras.models.load_model('model.h5')

# Open camera
cap = cv2.VideoCapture(0)

# Read video frame and transform to keras format
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Resize the video frame
    img = cv2.resize(frame, target_size)
    # Transform to numpy format
    input_arr = tf.keras.utils.img_to_array(img)
    input_arr = np.array([input_arr])

    # Use keras.Model.predict, send the model and input sample
    predictions =  tf.keras.Model.predict(model,x=input_arr)

    # Print the result of prediction, including class_index and confidence
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    text = 'Prediction: {}, Confidence: {:.2f}'.format(class_idx, confidence)
    cv2.putText(frame,text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video frame
    cv2.imshow('frame', frame)
    # Enter 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close camera
cv2.destroyAllWindows()