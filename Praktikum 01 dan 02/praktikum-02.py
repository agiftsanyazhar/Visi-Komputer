import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca Image
img = cv2.imread("Praktikum 01 dan 02/img/Logo Unair.png")

# ==============================
# Akses Piksel Citra
# ==============================
# Menampilkan ukuran dan struktur data Image
print("Ukuran Image ", img.shape)

# Get the height and width of the image
height, width = img.shape[0], img.shape[1]
print("Type Data ", img.dtype)

# Menampilkan Image
cv2.imshow("Image", img)

# Build an image , Content is zero filled
newImg1 = np.zeros((height, width, 3), np.uint8)
newImg2 = np.zeros((height, width, 3), np.uint8)
newImg3 = np.zeros((height, width, 3), np.uint8)

# Mengakses warna piksel BGR
print(img[136, 413])

# copy dan Set piksel BGR
newImg1 = img.copy()
newImg1[136:146, 413:423] = (0, 255, 0)

# Menampilkan Update Image
cv2.imshow("Image SetPixel", newImg1)
for i in range(height):
    for j in range(width):
        for k in range(3):  # Correspondence BGR Three channels
            newImg2[i, j][k] = img[i, j][k]

# ==============================
# Copy dan Flip Citra
# ==============================
# Menampilkan Copy Image
cv2.imshow("Image Copy", newImg2)
for i in range(height):
    for j in range(width):
        for k in range(3):  # Correspondence BGR Three channels
            newImg3[i, j][k] = img[height - 1 - i, j][k]

# Menampilkan Balik Image
cv2.imshow("Image Flip", newImg3)

# ==============================
# Flip the Images
# ==============================
imgFlipHorizontal = cv2.flip(img, 1)  # Flip horizontally
imgFlipVertical = cv2.flip(img, 0)  # Flip vertically
imgFlipBoth = cv2.flip(img, -1)  # Flip both horizontally and vertically

cv2.imshow("Image Flip Horizontal", imgFlipHorizontal)
cv2.imshow("Image Flip Vertical", imgFlipVertical)
cv2.imshow("Image Flip Both", imgFlipBoth)

# ==============================
# Flip the Images Using Matplotlib
# ==============================
# Display the images using matplotlib
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")

axes[1].imshow(cv2.cvtColor(imgFlipHorizontal, cv2.COLOR_BGR2RGB))
axes[1].set_title("Flip Horizontal")

axes[2].imshow(cv2.cvtColor(imgFlipVertical, cv2.COLOR_BGR2RGB))
axes[2].set_title("Flip Vertical")

axes[3].imshow(cv2.cvtColor(imgFlipBoth, cv2.COLOR_BGR2RGB))
axes[3].set_title("Flip Horizontal and Vertical")

plt.tight_layout()
plt.show()

# ==============================
# Rotate the Images
# ==============================
# Rotate the image clockwise 90 degrees
imgRotatedClockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Rotate the image counterclockwise 90 degrees
imgRotatedCounterclockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Rotate the image by an angle of 45 degrees using a custom rotation matrix
(cX, cY) = (img.shape[1] // 2, img.shape[0] // 2)  # Calculate the center of the image
angle = 45  # Rotation angle
scale = 1.0  # Scaling factor
rotationMatrix = cv2.getRotationMatrix2D((cX, cY), angle, scale)

# Apply the custom rotation using cv2.warpAffine()
imgRotatedCustom = cv2.warpAffine(img, rotationMatrix, (img.shape[1], img.shape[0]))

# Display the images using matplotlib
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Display the rotated image using matplotlib
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")

axes[1].imshow(cv2.cvtColor(imgRotatedClockwise, cv2.COLOR_BGR2RGB))
axes[1].set_title("Rotated Clockwise 90 Degrees")

axes[2].imshow(cv2.cvtColor(imgRotatedCounterclockwise, cv2.COLOR_BGR2RGB))
axes[2].set_title("Rotated Counterclockwise 90 Degrees")

axes[3].imshow(cv2.cvtColor(imgRotatedCustom, cv2.COLOR_BGR2RGB))
axes[3].set_title("Custom Rotated 45 Degrees")

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
