import tensorflow as tf 
import numpy as np
# model=tf.keras.models.load_model("waste_classifier.keras")
# class_name=[
#     "battery",
#     "biological",
#     "brown-glass",
#     "cardboard",
#     "clothes",
#     "green-glass",
#     "metal",
#     "paper",
#     "plastic",
#     "shoes",
#     "trash",
#     "white-glass"
# ]
# img_path=r"C:\Users\kumar\Downloads\archive (15)\garbage_dataset_split\train\battery\battery99.jpg"
# img=tf.keras.utils.load_img(
#     img_path,target_size=(224,224)
# )

# img_array=tf.keras.utils.img_to_array(img)
# img_array=tf.expand_dims(img_array,0)
# pred=model.predict(img_array)
# pre_idx=np.argmax(pred)
# pre_class=class_name[pre_idx]
# confidence=pred[0][pre_idx]*100
# print("predicated class:",pre_class)
# print(f"confidence:{confidence:.2f}%")

# import tensorflow as tf
# import numpy as np

# model = tf.keras.models.load_model("waste_classifier.keras")

# class_names = [
#     "battery",
#     "biological",
#     "brown-glass",
#     "cardboard",
#     "clothes",
#     "green-glass",
#     "metal",
#     "paper",
#     "plastic",
#     "shoes",
#     "trash",
#     "white-glass"
# ]

# image_path = r"C:\Users\kumar\Downloads\archive (15)\garbage_dataset_split\train\battery\battery99.jpg"

# img = tf.keras.utils.load_img(
#     image_path,
#     target_size=(224, 224)
# )

# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)


# predictions = model.predict(img_array, verbose=0)[0]

# top5 = np.argsort(predictions)[-5:][::-1]

# print("\nTop 5 predictions:")
