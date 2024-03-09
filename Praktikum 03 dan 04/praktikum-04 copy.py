import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Level 2,4, dan 8
# ==============================
# Read the picture
img = cv2.imread('Praktikum 03 dan 04/img/kucing.jpg')

# Get the height and width of the image
height, width = img.shape[0], img.shape[1]

# Build an image , Content is zero filled
new_img1 = np.zeros((height, width, 3), np.uint8)
new_img2 = np.zeros((height, width, 3), np.uint8)
new_img3 = np.zeros((height, width, 3), np.uint8)

# Image quantization operation, The quantification level is 2
for i in range(height):
    for j in range(width):
        for k in range(3): # Correspondence BGR Three channels
            if img[i, j][k] < 128:
                gray = 0
            else:
                gray = 129
            new_img1[i, j][k] = np.uint8(gray)

# Image quantization operation , The quantification level is 4
for i in range(height):
    for j in range(width):
        for k in range(3): # Correspondence BGR Three channels
            if img[i, j][k] < 64:
                gray = 0
            elif img[i, j][k] < 128:
                gray = 64
            elif img[i, j][k] < 192:
                gray = 128
            else:
                gray = 192
            new_img2[i, j][k] = np.uint8(gray)

# Image quantization operation , The quantification level is 8
for i in range(height):
    for j in range(width):
        for k in range(3): # Correspondence BGR Three channels
            if img[i, j][k] < 32:
                gray = 0
            elif img[i, j][k] < 64:
                gray = 32
            elif img[i, j][k] < 96:
                gray = 64
            elif img[i, j][k] < 128:
                gray = 96
            elif img[i, j][k] < 160:
                gray = 128
            elif img[i, j][k] < 192:
                gray = 160
            elif img[i, j][k] < 224:
                gray = 192
            else:
                gray = 224
        new_img3[i, j][k] = np.uint8(gray)

# Used to display Chinese tags normally
#plt.rcParams['font.sans-serif'] = ['SimHei']
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(new_img1, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(new_img2, cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(new_img3, cv2.COLOR_BGR2RGB)

# Show the image
titles = [
    u'(a) The original image ', u'(b) quantitative L2',
    u'(c) quantitative L4', u'(d) quantitative L8',
]

images = [img1, img2, img3, img4]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

# Wait for the display
cv2.waitKey(0)
cv2.destroyAllWindows()