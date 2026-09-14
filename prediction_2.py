import tensorflow as tf
import numpy as np

# Load your NEW 10-epoch model
model = tf.keras.models.load_model(r"C:\Users\kumar\OneDrive\Documents\datascience\waste_classifier_10epochs.keras")

# Class names — must be in the same order as training
class_names = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass"
]

# Image to test
image_path = r"C:\Users\kumar\Downloads\archive (15)\garbage_dataset_split\train\biological\biological96.jpg"
img=tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

# Convert image to array
img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension
img_array = tf.expand_dims(img_array, 0)

# IMPORTANT:
# Do NOT use mobilenet_v2.preprocess_input() here.
# It is already inside our saved model.

# Prediction
predictions = model.predict(img_array, verbose=0)[0]

# Get top 5 predictions
top5 = np.argsort(predictions)[-5:][::-1]

print("\nActual class: battery")
print("\nTop 5 predictions:")

for index in top5:
    print(
        f"{class_names[index]:15} "
        f"{predictions[index] * 100:.2f}%"
    )

# Best prediction
best_index = np.argmax(predictions)

print("\n-------------------------")
print("Predicted :", class_names[best_index])
print(f"Confidence: {predictions[best_index] * 100:.2f}%")
print("-------------------------")