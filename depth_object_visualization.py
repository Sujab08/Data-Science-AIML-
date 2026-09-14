import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load image
# -----------------------------

image_path = r"C:\Users\kumar\Downloads\archive (15)\garbage_classification_enhanced\battery\battery913.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

h, w = image.shape[:2]

# -----------------------------
# 2. GrabCut segmentation
# -----------------------------

mask = np.zeros((h, w), np.uint8)

bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

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

object_mask = np.where(
    (mask == 2) | (mask == 0),
    0,
    1
).astype(np.uint8)

# -----------------------------
# 3. Load depth model
# -----------------------------

depth_pipe = pipeline(
    "depth-estimation",
    model="Intel/dpt-hybrid-midas"
)

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(rgb)

# -----------------------------
# 4. Generate depth map
# -----------------------------

result = depth_pipe(pil_image)

depth = np.array(result["depth"])

depth = cv2.resize(
    depth,
    (w, h),
    interpolation=cv2.INTER_LINEAR
)

# -----------------------------
# 5. Keep only object depth
# -----------------------------

object_depth = depth.copy()

object_depth[object_mask == 0] = 0

# -----------------------------
# 6. Normalize object depth
# -----------------------------

values = object_depth[object_mask == 1]

min_depth = values.min()
max_depth = values.max()

normalized_depth = np.zeros_like(depth, dtype=np.uint8)

normalized_depth[object_mask == 1] = (
    (values - min_depth)
    / (max_depth - min_depth)
    * 255
).astype(np.uint8)

# -----------------------------
# 7. Print information
# -----------------------------

print("--------------------------------")
print("OBJECT DEPTH")
print("--------------------------------")

print("Object pixels :", len(values))
print("Minimum depth :", min_depth)
print("Maximum depth :", max_depth)
print("Mean depth    :", values.mean())
print("Median depth  :", np.median(values))

print("--------------------------------")

# -----------------------------
# 8. Display
# -----------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(normalized_depth, cmap="gray")
plt.title("Object Relative Depth")
plt.axis("off")

plt.show()