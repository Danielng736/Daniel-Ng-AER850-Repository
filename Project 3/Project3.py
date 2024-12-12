import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'C:/Users/Danie/OneDrive/Desktop/AER 815 Codes/Daniel-Ng-AER850-Repository/Project 3/Project 3 Data/motherboard_image.JPEG'
image = cv2.imread(img_path)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to smooth the image and reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred, 30, 150)

# Perform a dilation + erosion to close gaps between edge segments
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
dilated = cv2.dilate(edges, kernel, iterations=2)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Find contours from the thresholded image
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by area and then remove the small contours
contours = sorted(contours, key=cv2.contourArea, reverse=True)
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]  # Filter by area

# Create an empty mask to draw the contours
mask = np.zeros_like(gray)

# Draw the contours on the mask
for cnt in contours:
    cv2.drawContours(mask, [cnt], -1, color=255, thickness=cv2.FILLED)

result = cv2.bitwise_and(image, image, mask=mask)

plt.figure(figsize=(20, 10))

# Original image
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Canny edges
plt.subplot(1, 4, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

# Mask image
plt.subplot(1, 4, 3)
plt.imshow(mask, cmap='gray')
plt.title('Mask Image')
plt.axis('off')

# Extracted PCB
plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Extracted PCB')
plt.axis('off')

plt.show()