# Before run this script, you should complete the annotation and save it in pascal voc format
# An annotation software named labelImg: $ pip install labelImg
# The scripe will train and save a new model
# batch_size, epochs and validation_split depends on the number of pictures

import tensorflow as tf


# ----------Custom Inputs for Scripts----------
# Define some parameters for the loader
batch_size = 1
target_size = (640, 640)
# Point to the dataset location 
data_dir = 'pic_source'
# The number of classes
num_classes = 2
# Validation split
val_sp = 0.3
# ----------That's ALL You Need to Do----------


# Generate train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=val_sp,
  subset="training",
  seed=123,
  image_size=target_size,
  batch_size=batch_size)

# Generate validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=val_sp,
  subset="validation",
  seed=123,
  image_size=target_size,
  batch_size=batch_size)

# Standardize the data, make values to be in the [0,1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

# Train a model
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1
)

# Save the new trained model as keras format
model.save('model.h5')

# Access the loss and accuracy
loss, accuracy = model.evaluate(val_ds)
print(loss,accuracy)

print('Training work has Done!')