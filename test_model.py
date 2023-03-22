# Test your trained model(.h5) with test pictures(test dataset)

import tensorflow as tf
import numpy as np
from PIL import Image

# ----------Custom Inputs for Scripts----------
target_size = (640, 640)
# ----------That's ALL You Need to Do----------



# Load kreas model named xx.h5
model = tf.keras.models.load_model('model.h5')

# Load a test picture and resize
img = Image.open('test.jpg')
img = img.resize(target_size)

# Transform to numpy format
input_arr = tf.keras.utils.img_to_array(img)
input_arr = np.array([input_arr])

# Use keras.Model.predict, send the model and input sample
predictions =  tf.keras.Model.predict(model,x=input_arr)
class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx]
text = 'Prediction: {}, Confidence: {:.2f}'.format(class_idx, confidence)

print(text)

print('Model testing work has Done!')