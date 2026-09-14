import cv2
import numpy as np

image_path = r"C:\Users\kumar\Downloads\WhatsApp Image 2026-08-15 at 11.32.39 PM.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Could not load image")
    exit()

# Make a copy
result = image.copy()

# Create mask
mask = np.zeros(image.shape[:2], np.uint8)

# Background and foreground models
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Rectangle around the battery
# x, y, width, height
rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)

# GrabCut
cv2.grabCut(
    image,
    mask,
    rect,
    bgd_model,
    fgd_model,
    5,
    cv2.GC_INIT_WITH_RECT
)

# Convert mask
mask2 = np.where(
    (mask == 2) | (mask == 0),
    0,
    1
).astype("uint8")

# Extract foreground
foreground = image * mask2[:, :, np.newaxis]

# Find contours
gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    print("No object detected")
    exit()

# Largest contour
largest = max(contours, key=cv2.contourArea)

area = cv2.contourArea(largest)

x, y, width, height = cv2.boundingRect(largest)

# Draw bounding box
cv2.rectangle(
    result,
    (x, y),
    (x + width, y + height),
    (0, 255, 0),
    2
)

print("-------------------------")
print("Object detected!")
print("Area (pixels):", area)
print("Width (pixels):", width)
print("Height (pixels):", height)
print("-------------------------")

cv2.imshow("Detected Battery", result)
cv2.imshow("Foreground", foreground)

cv2.waitKey(0)
cv2.destroyAllWindows()