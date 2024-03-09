import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Kuantisasi Level 2
# ==============================
img = cv2.imread('Praktikum 03 dan 04/img/kucing.jpg')

height, width = img.shape[0], img.shape[1]
newImg = np.zeros((height, width, 3), np.uint8)

#The quantification level is 2
for i in range(height):
    for j in range(width):
        for k in range(3): # Correspondence BGR Three channels
            if img[i, j][k] < 128:
                gray = 0
            else:
                gray = 129
            newImg[i, j][k] = np.uint8(gray)

# Show the image
cv2.imshow('src', img)
cv2.imshow('new', newImg)

# ==============================
# Kuantisasi Citra Gray Level 1, 2, 3 dan 4
# ==============================
img = cv2.imread('Praktikum 03 dan 04/img/kucing.jpg')

img_gray = 128 * np.floor(img/128)
gray1=img_gray
img_gray = 64 * np.floor(img/64)
gray2= np.uint8(img_gray)
img_gray = 32 * np.floor(img/32)
gray3= np.uint8(img_gray)
img_gray = 16 * np.floor(img/16)
gray4= np.uint8(img_gray)

# Show the image
titles = [
    u'(a) Kuantisasi Gray L1', u'(b) Kuantisasi Gray L2',
    u'(c) Kuantisasi Gray L3', u'(d) Kuantisasi Gray L4',
]

images = [gray1, gray2, gray3, gray4]

for i in range(4):
    plt.subplot(2, 2, i+1),
    plt.imshow(images[i], cmap = 'gray', vmin = 0, vmax = 255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

# # ==============================
# # Level 2,4, dan 8
# # ==============================
# # Read the picture
# img = cv2.imread('Praktikum 03 dan 04/img/kucing.jpg')

# # Get the height and width of the image
# height, width = img.shape[0], img.shape[1]

# # Build an image , Content is zero filled
# new_img1 = np.zeros((height, width, 3), np.uint8)
# new_img2 = np.zeros((height, width, 3), np.uint8)
# new_img3 = np.zeros((height, width, 3), np.uint8)

# # Image quantization operation, The quantification level is 2
# for i in range(height):
#     for j in range(width):
#         for k in range(3): # Correspondence BGR Three channels
#             if img[i, j][k] < 128:
#                 gray = 0
#             else:
#                 gray = 129
#             new_img1[i, j][k] = np.uint8(gray)

# # Image quantization operation , The quantification level is 4
# for i in range(height):
#     for j in range(width):
#         for k in range(3): # Correspondence BGR Three channels
#             if img[i, j][k] < 64:
#                 gray = 0
#             elif img[i, j][k] < 128:
#                 gray = 64
#             elif img[i, j][k] < 192:
#                 gray = 128
#             else:
#                 gray = 192
#             new_img2[i, j][k] = np.uint8(gray)

# # Image quantization operation , The quantification level is 8
# for i in range(height):
#     for j in range(width):
#         for k in range(3): # Correspondence BGR Three channels
#             if img[i, j][k] < 32:
#                 gray = 0
#             elif img[i, j][k] < 64:
#                 gray = 32
#             elif img[i, j][k] < 96:
#                 gray = 64
#             elif img[i, j][k] < 128:
#                 gray = 96
#             elif img[i, j][k] < 160:
#                 gray = 128
#             elif img[i, j][k] < 192:
#                 gray = 160
#             elif img[i, j][k] < 224:
#                 gray = 192
#             else:
#                 gray = 224
#         new_img3[i, j][k] = np.uint8(gray)

# # Used to display Chinese tags normally
# #plt.rcParams['font.sans-serif'] = ['SimHei']
# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img2 = cv2.cvtColor(new_img1, cv2.COLOR_BGR2RGB)
# img3 = cv2.cvtColor(new_img2, cv2.COLOR_BGR2RGB)
# img4 = cv2.cvtColor(new_img3, cv2.COLOR_BGR2RGB)

# # Show the image
# titles = [
#     u'(a) The original image ', u'(b) quantitative L2',
#     u'(c) quantitative L4', u'(d) quantitative L8',
# ]

# images = [img1, img2, img3, img4]

# for i in range(4):
#     plt.subplot(2, 2, i+1), plt.imshow(images[i])
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])

# plt.show()

# ==============================
# Nomor 3
# ==============================
threshold_values = [100, 200, np.mean(img)]
titles = ["Threshold 100", "Threshold 200", f"Threshold Mean ({np.mean(img)})"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, threshold_value in enumerate(threshold_values):
    ret, thresholded_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    axes[i].imshow(thresholded_img, cmap="gray")
    axes[i].set_title(titles[i])
    axes[i].axis("off")

plt.show()

# ==============================
# Nomor 4
# ==============================
quantization_levels = [1, 2, 3, 4, 5, 6, 7, 8]
quantized_images = []

for level in quantization_levels:
    img_quantized = level * np.floor(img / level).astype(np.uint8)
    quantized_images.append(img_quantized)

titles_quantized = [f'Quantization Level {level}' for level in quantization_levels]

fig, axes = plt.subplots(2, 4, figsize=(15, 8))

for i in range(2):
    for j in range(4):
        index = i * 4 + j
        axes[i, j].imshow(quantized_images[index], cmap='gray', vmin=0, vmax=255)
        axes[i, j].set_title(titles_quantized[index])
        axes[i, j].axis("off")

plt.show()

# Wait for the display
cv2.waitKey(0)
cv2.destroyAllWindows()