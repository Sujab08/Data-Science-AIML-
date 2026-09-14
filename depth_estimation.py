import torch
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt

# Your image
image_path = r"C:\Users\kumar\Downloads\WhatsApp Image 2026-08-15 at 11.32.39 PM.jpeg"

# Load image
image = Image.open(image_path).convert("RGB")

# Load pretrained depth estimation model
depth_pipe = pipeline(
    task="depth-estimation",
    model="Intel/dpt-hybrid-midas"
)

# Estimate depth
result = depth_pipe(image)

# Depth map
depth = result["depth"]

# Display original image
plt.figure()
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Display estimated depth
plt.figure()
plt.imshow(depth, cmap="gray")
plt.title("Estimated Depth")
plt.axis("off")
plt.show()