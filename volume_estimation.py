import cv2

image_path = r"C:\Users\kumar\Downloads\archive (15)\garbage_classification_enhanced\battery\battery913.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image could not be loaded")
    exit()

# Display image
cv2.imshow("Waste Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()