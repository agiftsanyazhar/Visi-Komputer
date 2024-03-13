import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca data Image
images = [
    "img/kucing.jpg",
    "img/gradient.png",
    "img/Penguins.jpg",
    "img/arizona.jpg",
    "img/Tulips.jpg",
    "img/Koala.jpg",
    "img/pngtree-room-with-wood-floors-and-black-walls-from-a-3d-rendering-image_2611554.jpg",
    "img/underwater.jpg",
]

# Set up the subplot grid
rows = len(images)
cols = 4

# ==============================
# Histogram Pengaturan Brightness
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, 7, figsize=(15, 5 * rows))

# Loop through each image
for i, imgFilename in enumerate(images):
    # Membaca data Image
    oriImg = cv2.imread(imgFilename)

    # Display the original image
    axes[i, 0].imshow(cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")

    # Membaca data Image
    img = cv2.imread(imgFilename, 0)

    img1 = np.abs(img - 30.0)
    img1 = np.uint8(img1)
    img2 = img + 30.0
    img2 = np.uint8(img2)

    titles = [
        "Brightness -30",
        "Grayscale Image",
        "Brightness +30",
    ]

    brightnesImg = [img1, img, img2]

    for j in range(len(titles)):
        # Display the image
        axes[i, 2 * j + 1].imshow(brightnesImg[j], cmap="gray", vmin=0, vmax=255)
        axes[i, 2 * j + 1].set_title(f"{titles[j]}")
        axes[i, 2 * j + 1].axis("off")

        # Calculate and display the histogram
        hist = cv2.calcHist([brightnesImg[j]], [0], None, [256], [0, 256])
        axes[i, 2 * j + 2].plot(hist)
        axes[i, 2 * j + 2].set_title("Histogram")
        axes[i, 2 * j + 2].set_xlim([0, 256])

# Show the plot
plt.show()

# ==============================
# Histogram Pengaturan Contrast
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, 7, figsize=(15, 5 * rows))

# Loop through each image
for i, imgFilename in enumerate(images):
    # Membaca data Image
    oriImg = cv2.imread(imgFilename)

    # Display the original image
    axes[i, 0].imshow(cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")

    # Membaca data Image
    img = cv2.imread(imgFilename, 0)

    img1 = np.abs(img * 0.75)
    img1 = np.uint8(img1)
    img2 = np.abs(img * 1.25)
    img2 = np.uint8(img2)

    titles = [
        "Contrast x0.75",
        "Grayscale Image",
        "Contrast x1.25",
    ]

    contrastImg = [img1, img, img2]

    for j in range(len(titles)):
        # Display the image
        axes[i, 2 * j + 1].imshow(contrastImg[j], cmap="gray", vmin=0, vmax=255)
        axes[i, 2 * j + 1].set_title(f"{titles[j]}")
        axes[i, 2 * j + 1].axis("off")

        # Calculate and display the histogram
        hist = cv2.calcHist([contrastImg[j]], [0], None, [256], [0, 256])
        axes[i, 2 * j + 2].plot(hist)
        axes[i, 2 * j + 2].set_title("Histogram")
        axes[i, 2 * j + 2].set_xlim([0, 256])

# Show the plot
plt.show()

# ==============================
# Auto Level Gray Scale
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))

# Loop through each image
for i, imgFilename in enumerate(images):
    # Membaca data Image
    oriImg = cv2.imread(imgFilename)

    # Display the original image
    axes[i, 0].imshow(cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")

    # Membaca data Image
    img = cv2.imread(imgFilename)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x_min = np.min(img_gray)
    x_max = np.max(img_gray)

    print(f"For the image {imgFilename}:")
    print(f"x_min = {x_min}")
    print(f"x_max = {x_max}\n")

    sk = 255.0 / (x_max - x_min)
    img1 = sk * (img_gray - x_min)
    img1 = np.uint8(img1)

    titles = [
        "Grayscale Image",
        "Enhanced",
    ]

    enhancedImg = [img_gray, img1]

    for j in range(len(titles)):
        axes[i, j + 1].imshow(enhancedImg[j], cmap="gray", vmin=0, vmax=255)
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# Wait for the display
cv2.waitKey(0)
cv2.destroyAllWindows()
