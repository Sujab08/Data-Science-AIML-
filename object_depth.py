import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Load image
# -----------------------------------

image_path = r"C:\Users\kumar\Downloads\archive (15)\garbage_classification_enhanced\battery\battery913.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

# -----------------------------------
# 2. GrabCut segmentation
# -----------------------------------

mask = np.zeros(image.shape[:2], np.uint8)

bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

h, w = image.shape[:2]

# Initial rectangle
rect = (10, 10, w - 20, h - 20)

cv2.grabCut(
    image,
    mask,
    rect,
    bgd_model,
    fgd_model,
    5,
    cv2.GC_INIT_WITH_RECT
)

# Create binary object mask
object_mask = np.where(
    (mask == 2) | (mask == 0),
    0,
    1
).astype("uint8")

# -----------------------------------
# 3. Load depth model
# -----------------------------------

depth_pipe = pipeline(
    task="depth-estimation",
    model="Intel/dpt-hybrid-midas"
)

# Convert OpenCV image to PIL
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(rgb_image)

# -----------------------------------
# 4. Generate depth map
# -----------------------------------

result = depth_pipe(pil_image)

depth = np.array(result["depth"])

# Resize depth map to original image size
depth = cv2.resize(
    depth,
    (w, h)
)

# -----------------------------------
# 5. Get depth only inside object
# -----------------------------------

object_depth = depth[object_mask == 1]

if len(object_depth) == 0:
    print("No object detected!")
    exit()

# Calculate depth statistics
mean_depth = np.mean(object_depth)
median_depth = np.median(object_depth)
min_depth = np.min(object_depth)
max_depth = np.max(object_depth)

# -----------------------------------
# 6. Print results
# -----------------------------------

print("--------------------------------")
print("OBJECT DEPTH ANALYSIS")
print("--------------------------------")

print("Mean depth   :", mean_depth)
print("Median depth :", median_depth)
print("Minimum depth:", min_depth)
print("Maximum depth:", max_depth)

print("--------------------------------")

# -----------------------------------
# 7. Display results
# -----------------------------------

segmented = image * object_mask[:, :, np.newaxis]

plt.figure()
plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.title("Detected Object")
plt.axis("off")

plt.figure()
plt.imshow(depth, cmap="gray")
plt.title("Depth Map")
plt.axis("off")

plt.figure()
plt.imshow(object_mask, cmap="gray")
plt.title("Object Mask")
plt.axis("off")

plt.show()