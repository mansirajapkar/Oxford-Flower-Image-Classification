
#1. IMPORT REQUIRED LIBRARIES
import tensorflow as tf
import tensorflow_datasets as tfds
import scipy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
from keras import layers,models
#from tensorflow.keras import layers, models
print(tf.__version__)
#from tensorflow.keras import layers, models

print("TensorFlow Version:", tf.__version__)

# ===============================
# 2. LOAD OXFORD FLOWERS DATASET
# ===============================
print("\nLoading Oxford Flowers 102 dataset...")

(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'oxford_flowers102',
    split=['train[:80%]', 'train[80%:]', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

num_classes = ds_info.features['label'].num_classes
print("Number of Classes:", num_classes)

# ===============================
# 3. DATASET PARAMETERS
# ===============================
IMG_SIZE = 180
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# ===============================
# 4. PREPROCESS FUNCTION
# ===============================
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply preprocessing
ds_train = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)
ds_val   = ds_val.map(preprocess, num_parallel_calls=AUTOTUNE)
ds_test  = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE)

# Batch & Prefetch
ds_train = ds_train.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds_val   = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds_test  = ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ===============================
# 5. VISUALIZE SAMPLE IMAGES
# ===============================
print("\nDisplaying sample images...")

plt.figure(figsize=(10, 10))
for images, labels in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {labels[i].numpy()}")
        plt.axis("off")
plt.show()

# ===============================
# 6. DATA AUGMENTATION
# ===============================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ===============================
# 7. CNN MODEL
# ===============================
print("\nBuilding CNN Model...")

model = models.Sequential([
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes)
])

# ===============================
# 8. COMPILE MODEL
# ===============================
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# ===============================
# 9. TRAIN MODEL
# ===============================
EPOCHS = 15

print("\nTraining Model...")
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS
)

# ===============================
# 10. PLOT ACCURACY & LOSS
# ===============================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

# ===============================
# 11. EVALUATE MODEL
# ===============================
print("\nEvaluating on Test Dataset...")
test_loss, test_acc = model.evaluate(ds_test)
print("Test Accuracy:", test_acc)

# ===============================
# 12. SAVE MODEL
# ===============================
model.save("oxford_flower_model.keras")
print("\nModel saved as oxford_flower_model.keras")
