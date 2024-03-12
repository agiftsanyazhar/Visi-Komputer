import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca data Image
images = [
    "Praktikum 03 dan 04/img/kucing.jpg",
    "Praktikum 03 dan 04/img/gradient.png",
    "Praktikum 03 dan 04/img/Penguins.jpg",
    "Praktikum 03 dan 04/img/arizona.jpg",
    "Praktikum 03 dan 04/img/Tulips.jpg",
    "Praktikum 03 dan 04/img/Koala.jpg",
]

# Set up the subplot grid
rows = len(images)
cols = 4

# ==============================
# Kuantisasi Level 2
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))

# Loop through each image
for i, imgFilename in enumerate(images):
    # Membaca data Image
    img = cv2.imread(imgFilename)

    height, width = img.shape[0], img.shape[1]
    newImg = np.zeros((height, width, 3), np.uint8)

    # The quantification level is 2
    for x in range(height):
        for y in range(width):
            for k in range(3):  # Correspondence BGR Three channels
                if img[x, y][k] < 128:
                    gray = 0
                else:
                    gray = 129
                newImg[x, y][k] = np.uint8(gray)

    titles = [
        "Original Image",
        "Kuantisasi L2",
    ]

    quantizedImg = [img, newImg]

    for j in range(len(titles)):
        axes[i, j].imshow(cv2.cvtColor(quantizedImg[j], cv2.COLOR_BGR2RGB))
        axes[i, j].set_title(f"{titles[j]}")
        axes[i, j].axis("off")

# Show the plot
plt.show()

# ==============================
# Kuantisasi Citra Gray Level 1, 2, 3 dan 4
# ==============================
fig, axes = plt.subplots(rows, 6, figsize=(15, 5 * rows))

for i, imgFilename in enumerate(images):
    # Membaca data Image
    oriImg = cv2.imread(imgFilename)

    # Display the original image
    axes[i, 0].imshow(cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")

    # Membaca data Image
    img = cv2.imread(imgFilename, 0)

    imgGray = 128 * np.floor(img / 128)
    gray1 = imgGray
    imgGray = 64 * np.floor(img / 64)
    gray2 = np.uint8(imgGray)
    imgGray = 32 * np.floor(img / 32)
    gray3 = np.uint8(imgGray)
    imgGray = 16 * np.floor(img / 16)
    gray4 = np.uint8(imgGray)

    # Show the image
    titles = [
        "Grayscale Image",
        "Kuantisasi Gray L1",
        "Kuantisasi Gray L2",
        "Kuantisasi Gray L3",
        "Kuantisasi Gray L4",
    ]

    quantizedImg = [img, gray1, gray2, gray3, gray4]

    for j in range(len(titles)):
        axes[i, j + 1].imshow(quantizedImg[j], cmap="gray", vmin=0, vmax=255)
        axes[i, j + 1].set_title(titles[j])
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# ==============================
# Level 2,4, dan 8
# ==============================
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

for i, imgFilename in enumerate(images):
    # Membaca data Image
    img = cv2.imread(imgFilename)

    # Get the height and width of the image
    height, width = img.shape[0], img.shape[1]

    # Build an image , Content is zero filled
    new_img1 = np.zeros((height, width, 3), np.uint8)
    new_img2 = np.zeros((height, width, 3), np.uint8)
    new_img3 = np.zeros((height, width, 3), np.uint8)

    # Image quantization operation, The quantification level is 2
    for x in range(height):
        for y in range(width):
            for z in range(3):  # Correspondence BGR Three channels
                if img[x, y][z] < 128:
                    gray = 0
                else:
                    gray = 129
                new_img1[x, y][z] = np.uint8(gray)

    # Image quantization operation , The quantification level is 4
    for x in range(height):
        for y in range(width):
            for z in range(3):  # Correspondence BGR Three channels
                if img[x, y][z] < 64:
                    gray = 0
                elif img[x, y][z] < 128:
                    gray = 64
                elif img[x, y][z] < 192:
                    gray = 128
                else:
                    gray = 192
                new_img2[x, y][z] = np.uint8(gray)

    # Image quantization operation , The quantification level is 8
    for x in range(height):
        for y in range(width):
            for z in range(3):  # Correspondence BGR Three channels
                if img[x, y][z] < 32:
                    gray = 0
                elif img[x, y][z] < 64:
                    gray = 32
                elif img[x, y][z] < 96:
                    gray = 64
                elif img[x, y][z] < 128:
                    gray = 96
                elif img[x, y][z] < 160:
                    gray = 128
                elif img[x, y][z] < 192:
                    gray = 160
                elif img[x, y][z] < 224:
                    gray = 192
                else:
                    gray = 224
                new_img3[x, y][z] = np.uint8(gray)

    # Used to display Chinese tags normally
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(new_img1, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(new_img2, cv2.COLOR_BGR2RGB)
    img4 = cv2.cvtColor(new_img3, cv2.COLOR_BGR2RGB)

    # Show the image
    titles = [
        "Original image ",
        "Kuantisasi L2",
        "Kuantisasi L4",
        "Kuantisasi L8",
    ]

    quantizedImg = [img1, img2, img3, img4]

    for j in range(len(titles)):
        axes[i, j].imshow(quantizedImg[j])
        axes[i, j].set_title(titles[j])
        axes[i, j].axis("off")

# Show the plot
plt.show()

# ==============================
# Nomor 3
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

# Loop through each image
for i, imgFilename in enumerate(images):
    # Membaca data Image
    img = cv2.imread(imgFilename)

    # Display the original image
    axes[i, 0].imshow(img, cmap="gray")
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")

    img = cv2.imread(imgFilename, 0)

    titles = ["Threshold", "Threshold", "Threshold Mean"]

    # Threshold values
    thresholdValues = [100, 200, np.mean(img)]

    for j, threshold in enumerate(thresholdValues):
        # Apply thresholding
        ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        # Display the images
        axes[i, j + 1].imshow(thresh, cmap="gray")
        axes[i, j + 1].set_title(f"{titles[j]}: {round(threshold, 3)}")
        axes[i, j + 1].axis("off")

# ==============================
# Nomor 4
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, 9, figsize=(15, 5 * rows))

# Loop through each image
for i, imgFilename in enumerate(images):
    # Membaca data Image
    img = cv2.imread(imgFilename)

    # Display the original image
    axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")

    # Quantization levels
    quantizationLevels = [1, 2, 3, 4, 5, 6, 7, 8]

    for j, level in enumerate(quantizationLevels):
        # Calculate the step size for the current quantization level
        step = 255 / (level if level > 0 else 1)

        # Apply quantization
        quantizedImg = ((img / step).astype(np.uint8) * step).astype(np.uint8)

        # Display the quantized image
        axes[i, j + 1].imshow(cv2.cvtColor(quantizedImg, cv2.COLOR_BGR2RGB))
        axes[i, j + 1].set_title(f"Q{level}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# Wait for the display
cv2.waitKey(0)
cv2.destroyAllWindows()
