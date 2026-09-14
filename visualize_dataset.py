import os
import random
import matplotlib.pyplot as plt
from PIL import Image
data_path=r"C:\Users\kumar\Downloads\archive (15)\garbage_classification_enhanced"
# classes = sorted([
#     folder for folder in os.listdir(data_path)
#     if os.path.isdir(os.path.join(data_path, folder))
# ])

# plt.figure(figsize=(15, 10))

# for i, class_name in enumerate(classes):
#     class_path = os.path.join(data_path, class_name)

#     images = [
#         f for f in os.listdir(class_path)
#         if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
#     ]

#     image_file = random.choice(images)
#     image_path = os.path.join(class_path, image_file)

#     image = Image.open(image_path)

#     plt.subplot(3, 4, i + 1)
#     plt.imshow(image)
#     plt.title(class_name)
#     plt.axis("off")

# plt.tight_layout()
# plt.show()


# from collections import Counter
# sizes=Counter()
# for class_name in os.listdir(data_path):
#     class_path=os.path.join(data_path,class_name)
#     if not os.path.isdir(class_path):
#         continue
#     for file in os.listdir(class_path):
#         if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
#             image_path=os.path.join(class_path,file)

#             try:
#                 with Image.open(image_path) as img:
#                     sizes[img.size] +=1
#             except Exception:
#                 print("Problem:",image_path)

# print("different imahe sizes:",len(sizes))
# print("\n most commeon sizes:")
# for size,count in sizes.most_common(20):
#     print(size,':',count)


# total=0
# valid=0
# corrupt=0
# for class_name in os.listdir(data_path):
#     class_path=os.path.join(data_path,class_name)

#     if not os.path.isdir(class_path):
#         continue

#     for file in os.listdir(class_path):
#         if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
#             total+=1
#             image_path=os.path.join(class_path,file)

#             try:
#                 with Image.open(image_path) as img:
#                     img.verify()
#                 valid+=1
#             except Exception:
#                 corrupt+=1
#                 print("currupted:",image_path)
# print('\n--------')
# print('total images:',total)
# print('valid images:',valid)
# print('corrupt images:',corrupt)

# import shutil
# from sklearn.model_selection import train_test_split
# dataset_path = r"C:\Users\kumar\Downloads\archive (15)\garbage_classification_enhanced"

# output_path = r"C:\Users\kumar\Downloads\archive (15)\garbage_dataset_split"
# classes = [
#     folder for folder in os.listdir(dataset_path)
#     if os.path.isdir(os.path.join(dataset_path, folder))
# ]

# for class_name in classes:

#     class_path = os.path.join(dataset_path, class_name)

#     images = [
#         file for file in os.listdir(class_path)
#         if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
#     ]

#     # 70% train, 30% temporary
#     train_images, temp_images = train_test_split(
#         images,
#         test_size=0.30,
#         random_state=42
#     )

#     # Split remaining 30% into 15% validation + 15% test
#     val_images, test_images = train_test_split(
#         temp_images,
#         test_size=0.50,
#         random_state=42
#     )

#     # Create folders
#     for split in ["train", "validation", "test"]:
#         os.makedirs(
#             os.path.join(output_path, split, class_name),
#             exist_ok=True
#         )

#     # Copy images
#     for image in train_images:
#         shutil.copy2(
#             os.path.join(class_path, image),
#             os.path.join(output_path, "train", class_name, image)
#         )

#     for image in val_images:
#         shutil.copy2(
#             os.path.join(class_path, image),
#             os.path.join(output_path, "validation", class_name, image)
#         )

#     for image in test_images:
#         shutil.copy2(
#             os.path.join(class_path, image),
#             os.path.join(output_path, "test", class_name, image)
#         )

#     print(
#         class_name,
#         "→ Train:", len(train_images),
#         "Validation:", len(val_images),
#         "Test:", len(test_images)
#     )
# print("\nDataset split completed!")

import os

dataset_path = r"C:\Users\kumar\Downloads\archive (15)\garbage_dataset_split"

for split in ["train", "validation", "test"]:
    print(f"\n{split.upper()}")

    total = 0

    for class_name in sorted(os.listdir(os.path.join(dataset_path, split))):
        class_path = os.path.join(dataset_path, split, class_name)

        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ])

            total += count
            print(f"{class_name:15}: {count}")

    print("Total:", total)