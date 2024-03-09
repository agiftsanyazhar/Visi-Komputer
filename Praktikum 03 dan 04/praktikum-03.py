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
# Layer B/G/R
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

# Loop through each image
for i, imgFilename in enumerate(images):
    # Membaca data Image
    img = cv2.imread(imgFilename)

    # Extract RGB, B, G, R layers
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    b = img.copy()
    b[:, :, 1] = 0
    b[:, :, 2] = 0

    g = img.copy()
    g[:, :, 0] = 0
    g[:, :, 2] = 0

    r = img.copy()
    r[:, :, 0] = 0
    r[:, :, 1] = 0

    titles = [
        "Original Image",
        "B-RGB",
        "G-RGB",
        "R-RGB",
    ]

    layeredImages = [img, b, g, r]

    for j in range(len(titles)):
        axes[i, j].imshow(cv2.cvtColor(layeredImages[j], cv2.COLOR_BGR2RGB))
        axes[i, j].set_title(f"Image {j + 1} ({titles[j]})")
        axes[i, j].axis("off")

# Show the plot
plt.show()

# ==============================
# Grayscale dari B/G/R
# ==============================
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

for i, imgFilename in enumerate(images):
    # Membaca data Image
    img = cv2.imread(imgFilename)

    # Extract RGB, B, G, R layers
    B, G, R = cv2.split(img)

    titles = [
        "Original Image",
        "Channel B",
        "Channel G",
        "Channel R",
    ]

    grayscaledImages = [img, B, G, R]

    for j in range(len(titles)):
        axes[i, j].imshow(cv2.cvtColor(grayscaledImages[j], cv2.COLOR_BGR2RGB))
        axes[i, j].set_title(f"Image {j + 1} ({titles[j]})")
        axes[i, j].axis("off")

# Show the plot
plt.show()

# ==============================
# Grayscale dengan Iluminasi Citra
# ==============================
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

for i, imgFilename in enumerate(images):
    # Membaca data Image
    img = cv2.imread(imgFilename)

    # Extract RGB, B, G, R layers
    B, G, R = cv2.split(img)

    imgGray1 = 0.33 * R + 0.33 * G + 0.33 * B
    imgGray1 = imgGray1.astype(np.uint8)
    imgRG1 = np.minimum(R, G)
    imgGray2 = np.minimum(imgRG1, B)
    imgRG2 = np.maximum(R, G)
    imgGray3 = np.maximum(imgRG2, B)

    titles = [
        "Original Image",
        "Iluminasi Rata-Rata",
        "Iluminasi Minimum",
        "Iluminasi Maksimum",
    ]

    illuminatedImages = [img, imgGray1, imgGray2, imgGray3]

    for j in range(len(titles)):
        axes[i, j].imshow(cv2.cvtColor(illuminatedImages[j], cv2.COLOR_BGR2RGB))
        axes[i, j].set_title(f"Image {j + 1} ({titles[j]})")
        axes[i, j].axis("off")

# Show the plot
plt.show()

# ==============================
# Citra Binary dengan Parameter Threshold
# ==============================
fig, axes = plt.subplots(rows, 6, figsize=(15, 5 * rows))

for i, imgFilename in enumerate(images):
    # Membaca data Image
    img = cv2.imread(imgFilename, 0)

    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = [
        "Original Image",
        "BINARY",
        "BINARY_INV",
        "TRUNC",
        "TOZERO",
        "TOZERO_INV",
    ]

    thresholded_images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for j in range(len(titles)):
        axes[i, j].imshow(cv2.cvtColor(thresholded_images[j], cv2.COLOR_BGR2RGB))
        axes[i, j].set_title(titles[j])
        axes[i, j].axis("off")

# Show the plot
plt.show()

# ==============================
# Nomor 1
# ==============================
fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))

for i, imgFilename in enumerate(images):
    # Membaca data Image
    img = cv2.imread(imgFilename)

    # Convert the image to RGB for displaying with matplotlib
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Sepia Effect
    sepiaImg = imgRgb.copy()
    sepiaImg[:, :, 0] = 2 * sepiaImg[:, :, 2]  # R channel
    sepiaImg[:, :, 1] = 1.8 * sepiaImg[:, :, 2]  # G channel
    sepiaImg[:, :, 2] = sepiaImg[:, :, 2]  # B channel

    titles = [
        "Original Image",
        "Sepia Effect",
    ]

    sepiazedImg = [imgRgb, sepiaImg]

    for j in range(len(titles)):
        axes[i, j].imshow(sepiazedImg[j])
        axes[i, j].set_title(titles[j])
        axes[i, j].axis("off")

# Show the plot
plt.show()

# ==============================
# Nomor 4
# ==============================
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

for i, imgFilename in enumerate(images):
    # Membaca data Image
    img = cv2.imread(imgFilename, cv2.IMREAD_GRAYSCALE)

    imgBlur = cv2.medianBlur(img, 5)

    ret, th1 = cv2.threshold(imgBlur, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    titles = [
        'Original Image', 'Global Thresholding (v = 127)',
        'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding',
    ]
    images = [img, th1, th2, th3]

    for j in range(len(titles)):
        axes[i, j].imshow(images[j], 'gray')
        axes[i, j].set_title(titles[j])
        axes[i, j].axis("off")

# Show the plot
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
