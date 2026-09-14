import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

# ==========================================
# 1. IMAGE
# ==========================================

image_path = r"C:\Users\kumar\Downloads\archive (15)\garbage_classification_enhanced\battery\battery913.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

h, w = image.shape[:2]

# ==========================================
# 2. KNOWN REAL DIMENSIONS
# ==========================================

real_width_cm = 25
real_height_cm = 30
real_depth_cm = 17

# ==========================================
# 3. GRABCUT OBJECT SEGMENTATION
# ==========================================

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

# ==========================================
# 4. FIND OBJECT DIMENSIONS IN PIXELS
# ==========================================

contours, _ = cv2.findContours(
    object_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    print("Object not detected!")
    exit()

largest = max(contours, key=cv2.contourArea)

x, y, pixel_width, pixel_height = cv2.boundingRect(largest)

print("--------------------------------")
print("OBJECT DETECTION")
print("--------------------------------")

print("Pixel width  :", pixel_width)
print("Pixel height :", pixel_height)

# ==========================================
# 5. PIXEL → CM CALIBRATION
# ==========================================

width_scale = pixel_width / real_width_cm
height_scale = pixel_height / real_height_cm

pixels_per_cm = (width_scale + height_scale) / 2

print("--------------------------------")
print("SCALE CALIBRATION")
print("--------------------------------")

print(f"Width scale  : {width_scale:.2f} pixels/cm")
print(f"Height scale : {height_scale:.2f} pixels/cm")
print(f"Average scale: {pixels_per_cm:.2f} pixels/cm")

# ==========================================
# 6. DEPTH MODEL
# ==========================================

depth_pipe = pipeline(
    "depth-estimation",
    model="Intel/dpt-hybrid-midas"
)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pil_image = Image.fromarray(rgb_image)

depth_result = depth_pipe(pil_image)

depth = np.array(depth_result["depth"])

depth = cv2.resize(
    depth,
    (w, h),
    interpolation=cv2.INTER_LINEAR
)

# ==========================================
# 7. OBJECT DEPTH
# ==========================================

object_depth = depth[object_mask == 1]

if len(object_depth) == 0:
    print("No depth information found!")
    exit()

mean_depth = np.mean(object_depth)
median_depth = np.median(object_depth)

print("--------------------------------")
print("DEPTH INFORMATION")
print("--------------------------------")

print(f"Mean relative depth   : {mean_depth:.2f}")
print(f"Median relative depth : {median_depth:.2f}")

# ==========================================
# 8. CURRENT CALIBRATED VOLUME
# ==========================================

volume_cm3 = (
    real_width_cm *
    real_height_cm *
    real_depth_cm
)

volume_liters = volume_cm3 / 1000

print("--------------------------------")
print("VOLUME")
print("--------------------------------")

print(f"Width  : {real_width_cm} cm")
print(f"Height : {real_height_cm} cm")
print(f"Depth  : {real_depth_cm} cm")

print(f"Volume : {volume_cm3:.2f} cm³")
print(f"Volume : {volume_liters:.2f} L")

print("--------------------------------")