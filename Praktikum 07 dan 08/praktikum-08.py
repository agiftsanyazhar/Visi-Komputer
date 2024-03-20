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
]

# Set up the subplot grid
rows = len(images)
cols = 4

# ==============================
# Filtering Convolusi dengan Beberapa Kernel
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, 6, figsize=(15, 5 * rows))

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

    kernel = np.ones((15, 15), np.float32) / 225
    dst_blur = cv2.filter2D(img, -1, kernel)
    kernel_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_h = np.transpose(kernel_v)
    dst_h = cv2.filter2D(img, -1, kernel_h)
    dst_v = cv2.filter2D(img, -1, kernel_v)
    dst_edge = cv2.add(dst_v, dst_h)
    img2 = cv2.add(img, dst_edge)
    img_inverse = 255 - dst_edge

    titles = [
        "Grayscale Image",
        "Filtered Blur",
        "Filtered Edge",
        "Filtered Sharpeness",
        "Filtered Sketsa",
    ]

    brightnesImg = [img, dst_blur, dst_edge, img2, img_inverse]

    for j in range(len(titles)):
        # Display the image
        axes[i, j + 1].imshow(brightnesImg[j], cmap="gray")
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.tight_layout()
plt.show()

# ==============================
# Filter Blur dengan Berbagai Ukuran Mask
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, 5, figsize=(15, 5 * rows))

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

    img_blur1 = cv2.blur(img, [5, 5])
    img_blur2 = cv2.blur(img, [11, 11])
    img_blur3 = cv2.blur(img, [25, 25])

    titles = [
        "Grayscale Image",
        "Blur 5x5",
        "Blur 11x11",
        "Blur 25x25",
    ]

    brightnesImg = [img, img_blur1, img_blur2, img_blur3]

    for j in range(len(titles)):
        # Display the image
        axes[i, j + 1].imshow(brightnesImg[j], cmap="gray")
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.tight_layout()
plt.show()

# ==============================
# Citra + Noise
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

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

    noise = np.random.randn(*img.shape) * 10
    img_noise = np.abs(img + noise)
    img_noise = np.uint8(img_noise)
    noise = np.random.randn(*img.shape) * 30
    img_noise = np.abs(img + noise)
    img_noise_2 = np.uint8(img_noise)

    titles = [
        "Grayscale Image",
        "Image Noise (x10)",
        "Image Noise (x30)",
    ]

    brightnesImg = [img, img_noise, img_noise_2]

    for j in range(len(titles)):
        # Display the image
        axes[i, j + 1].imshow(brightnesImg[j], cmap="gray")
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.tight_layout()
plt.show()

# ==============================
# Reduksi Noise Grausian dan Median Filter
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, 5, figsize=(15, 5 * rows))

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

    noise = np.random.randn(*img.shape) * 30
    img_noise = np.abs(img + noise)
    img_noise = np.uint8(img_noise)
    dst_gauss = cv2.GaussianBlur(img_noise, [7, 7], 0)
    dst_med = cv2.medianBlur(img_noise, 7)

    titles = [
        "Grayscale Image",
        "Image Noise (x30)",
        "Gaussian",
        "Median",
    ]

    brightnesImg = [img, img_noise, dst_gauss, dst_med]

    for j in range(len(titles)):
        # Display the image
        axes[i, j + 1].imshow(brightnesImg[j], cmap="gray")
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.tight_layout()
plt.show()

# ==============================
# Deteksi Tepi Sobel, Canny dan Laplacian Filter
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, 5, figsize=(15, 5 * rows))

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

    sobel_x = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobel_y = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobel = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    sobel = cv2.convertScaleAbs(sobel)

    canny = cv2.Canny(img, 100, 200)
    im_lap = cv2.Laplacian(img, ksize=5, ddepth=cv2.CV_64F)
    im_lap = cv2.convertScaleAbs(im_lap)

    titles = [
        "Grayscale Image",
        "Sobel",
        "Canny",
        "Laplacian",
    ]

    brightnesImg = [img, sobel, canny, im_lap]

    for j in range(len(titles)):
        # Display the image
        axes[i, j + 1].imshow(brightnesImg[j], cmap="gray")
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.tight_layout()
plt.show()

# ==============================
# Sharpeness
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

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

    kernel1 = np.ones([5, 5], np.float32) / 25
    img_blur = cv2.filter2D(img, -1, kernel1)
    kernel2a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel2b = np.transpose(kernel2a)
    kernel2 = kernel2a + kernel2b
    img_edge = 0.5 * cv2.filter2D(img, -1, kernel2)
    img_edge = np.uint8(img_edge)
    img_sharp = cv2.add(img_blur, img_edge)

    kernel1 = np.ones([15, 15], np.float32) / 225
    img_blur = cv2.filter2D(img, -1, kernel1)
    kernel2a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel2b = np.transpose(kernel2a)
    kernel2 = kernel2a + kernel2b
    img_edge = 0.5 * cv2.filter2D(img, -1, kernel2)
    img_edge = np.uint8(img_edge)
    img_sharp2 = cv2.add(img_blur, img_edge)

    titles = [
        "Grayscale Image",
        "Sharpeness (5x5)",
        "Sharpeness (15x15)",
    ]

    brightnesImg = [img, img_sharp, img_sharp2]

    for j in range(len(titles)):
        # Display the image
        axes[i, j + 1].imshow(brightnesImg[j], cmap="gray")
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.tight_layout()
plt.show()

# ==============================
# Citra Sketsa
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
    img = cv2.imread(imgFilename, 0)

    kernel = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]])
    dst = cv2.filter2D(img, -1, kernel)
    sketsa = 255 - dst

    titles = [
        "Grayscale Image",
        "Sketsa",
    ]

    brightnesImg = [img, sketsa]

    for j in range(len(titles)):
        # Display the image
        axes[i, j + 1].imshow(brightnesImg[j], cmap="gray")
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.tight_layout()
plt.show()

# Wait for the display
cv2.waitKey(0)
cv2.destroyAllWindows()
