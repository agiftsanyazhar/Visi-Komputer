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
]

# Set up the subplot grid
rows = len(images)
cols = 4

# ==============================
# Brightness
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

    img1 = np.abs(img - 30.0)
    img1 = np.uint8(img1)
    img2 = img + 30.0
    img2 = np.uint8(img2)

    titles = [
        "Grayscale Image",
        "Brightness -30",
        "Brightness +30",
    ]

    brightnesImg = [img, img1, img2]

    for j in range(len(titles)):
        axes[i, j + 1].imshow(brightnesImg[j], cmap="gray", vmin=0, vmax=255)
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# ==============================
# Contrast
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
    img = cv2.imread(imgFilename)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img1 = imgGray * 0.5
    img2 = imgGray * 0.75
    img3 = imgGray * 1.25
    img4 = imgGray * 1.5

    titles = [
        "Contrast x0.5",
        "Contrast x0.75",
        "Original Image",
        "Contrast x1.25",
        "Contrast x1.5",
    ]

    contrastImg = [img1, img2, imgGray, img3, img4]

    for j in range(len(titles)):
        axes[i, j + 1].imshow(contrastImg[j], cmap="gray", vmin=0, vmax=255)
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# ==============================
# Invers
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

    img1 = 255 - img

    titles = [
        "Original Image",
        "Invers",
    ]

    inverseImg = [img, img1]

    for j in range(len(titles)):
        axes[i, j + 1].imshow(inverseImg[j], cmap="gray", vmin=0, vmax=255)
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# ==============================
# Transformasi Exponensial
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

    img = img / 255
    img1 = 100.0 * np.exp(0.5 * img)
    img1 = np.uint8(img1)
    img2 = 50.0 * np.exp(2.0 * img)
    img2 = np.uint8(img2)

    titles = [
        "Original Image",
        "a = 100; b = 0.5",
        "a = 50; b = 2",
    ]

    transformasiImg = [img, img1, img2]

    for j in range(len(titles)):
        axes[i, j + 1].imshow(transformasiImg[j], cmap="gray")
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# ==============================
# Logaritmik
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

    epsilon = 1e-10  # Small constant to avoid division by zero

    img1 = 40 * np.log(0.5 * (img + epsilon))
    img1 = np.uint8(img1)
    img2 = 40.0 * np.log(2.0 * (img + epsilon))
    img2 = np.uint8(img2)

    titles = [
        "Original Image",
        "a = 40; b = 0.5",
        "a = 40; b = 2",
    ]

    transformasiImg = [img, img1, img2]

    for j in range(len(titles)):
        axes[i, j + 1].imshow(transformasiImg[j], cmap="gray")
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# ==============================
# 2. Apa perbedaan antara brightness dan contrast?
#       Brightness (Kecerahan): Merujuk pada keseluruhan kecerahan atau kegelapan suatu gambar.
#                               Meningkatkan kecerahan membuat gambar secara menjadi lebih terang,
#                               sementara mengurangi kecerahan membuatnya menjadi lebih gelap.
#       Contrast (Kontras): Merujuk pada perbedaan intensitas antara bagian tergelap dan tercerah
#                           dari suatu gambar. Meningkatkan kontras membuat area terang menjadi
#                           lebih terang dan area gelap menjadi lebih gelap, sementara mengurangi
#                           kontras mengurangi perbedaan ini.
# ==============================

# ==============================
# Nomor 3
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

    img1 = np.abs(img - 30.0)
    img1 = np.uint8(img1)
    img2 = img + 30.0
    img2 = np.uint8(img2)
    imgBright = img + 255.0
    imgBright = np.clip(imgBright, 0, 255)  # Ensure values are within 0-255
    imgBright = np.uint8(imgBright)

    titles = [
        "Grayscale Image",
        "Brightness -30",
        "Brightness +30",
        "Brightness +255",
    ]

    brightnesImg = [img, img1, img2, imgBright]

    for j in range(len(titles)):
        axes[i, j + 1].imshow(brightnesImg[j], cmap="gray", vmin=0, vmax=255)
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# ==============================
# Nomor 4 dan 5
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

    img1 = 255 - img

    # Menerapkan rumus invers xb=128-xg
    imgInverse = 128 - img
    imgInverse = np.clip(
        imgInverse, 0, 255
    )  # Pastikan nilai berada dalam rentang 0-255
    imgInverse = np.uint8(imgInverse)

    titles = [
        "Original Image",
        "Invers",
        "128 - xg",
    ]

    inverseImg = [img, img1, imgInverse]

    for j in range(len(titles)):
        axes[i, j + 1].imshow(inverseImg[j], cmap="gray", vmin=0, vmax=255)
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# ==============================
# Nomor 6
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, 5, figsize=(15, 5 * rows))

# Loop through each image
for i, imgFilename in enumerate(images):
    # Membaca data Image
    img = cv2.imread(imgFilename, 0)

    # Normalisasi piksel gambar ke rentang [0,1]
    imgNormalized = img / 255.0

    # Lakukan transformasi power dengan beberapa nilai gamma
    gammaValues = [0.5, 1.0, 1.5, 2.0]
    powertransformedImages = [np.power(imgNormalized, gamma) for gamma in gammaValues]

    # Konversi kembali gambar ke rentang piksel [0,255]
    powertransformedImages = [np.uint8(img * 255) for img in powertransformedImages]

    # Tampilkan gambar asli dan gambar yang telah ditransformasi
    titles = ["Original Image"] + [f"Gamma = {gamma}" for gamma in gammaValues]
    imagesToShow = [img] + powertransformedImages

    for j in range(len(titles)):
        axes[i, j].imshow(imagesToShow[j], cmap="gray", vmin=0, vmax=255)
        axes[i, j].set_title(f"{titles[j]}")
        axes[i, j].axis("off")

# Show the plot
plt.show()

# ==============================
# Nomor 7
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

    # Hitung transformasi Fourier dari gambar
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Hitung magnitudo spektrum Fourier
    magnitude_spectrum = 20 * np.log1p(np.abs(fshift))

    # Lakukan transformasi logaritmik pada spektrum Fourier
    c = 10.0  # Parameter untuk enhancement
    log_transformed = c * np.log1p(
        np.abs(magnitude_spectrum)
    )  # Use np.log1p to avoid log(0)

    # Lakukan transformasi logaritmik dengan enhancement
    enhanced_spectrum = c * np.log1p(np.abs(fshift))  # Use np.log1p to avoid log(0)

    # Normalisasi spektrum Fourier ke rentang [0,255]
    enhanced_spectrum_normalized = (
        enhanced_spectrum / np.max(enhanced_spectrum)
    ) * 255.0
    enhanced_spectrum_normalized = np.uint8(enhanced_spectrum_normalized)

    # Tampilkan citra spektrum Fourier dan hasil enhancement
    titles = ["Fourier Spectrum", f"Enhanced Spectrum (c = {c})"]
    images_to_show = [log_transformed, enhanced_spectrum_normalized]

    for j in range(len(titles)):
        axes[i, j + 1].imshow(images_to_show[j], cmap="gray")
        axes[i, j + 1].set_title(titles[j])
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# ==============================
# Nomor 8
# ==============================
# Create a subplot for each image
fig, axes = plt.subplots(rows, 8, figsize=(15, 5 * rows))

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

    epsilon = 1e-10  # Small constant to avoid division by zero

    img1 = 40 * np.log(0.5 * (img + epsilon))
    img1 = np.uint8(img1)
    img2 = 40.0 * np.log(2.0 * (img + epsilon))
    img2 = np.uint8(img2)

    # Invers Log
    invers_log_img1 = np.exp(img1 / 40.0) - epsilon
    invers_log_img2 = np.exp(img2 / 40.0) - epsilon

    # Konversi ke rentang piksel [0, 255]
    invers_log_img1 = (invers_log_img1 / np.max(invers_log_img1)) * 255.0
    invers_log_img2 = (invers_log_img2 / np.max(invers_log_img2)) * 255.0

    # Ubah tipe data ke uint8
    invers_log_img1 = np.uint8(invers_log_img1)
    invers_log_img2 = np.uint8(invers_log_img2)

    # Root
    root_img1 = np.power(img1 / 40.0, 2.5)
    root_img2 = np.power(img2 / 40.0, 0.5)

    # Konversi ke rentang piksel [0, 255]
    root_img1 = (root_img1 / np.max(root_img1)) * 255.0
    root_img2 = (root_img2 / np.max(root_img2)) * 255.0

    # Ubah tipe data ke uint8
    root_img1 = np.uint8(root_img1)
    root_img2 = np.uint8(root_img2)

    titles = [
        "Original Image",
        "a = 40; b = 0.5",
        "a = 40; b = 2",
        "Invers Log 1",
        "Invers Log 2",
        "Root Img 1",
        "Root Img 2",
    ]

    transformasiImg = [
        img,
        img1,
        img2,
        invers_log_img1,
        invers_log_img2,
        root_img1,
        root_img2,
    ]

    for j in range(len(titles)):
        axes[i, j + 1].imshow(transformasiImg[j], cmap="gray")
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.show()

# Wait for the display
cv2.waitKey(0)
cv2.destroyAllWindows()
