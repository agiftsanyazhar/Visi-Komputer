import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, color, io
import cv2
import matplotlib.pyplot as plt

# load image from images directory
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

# Create a subplot for each image
fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))

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

    # Convert the image to a 2D array (graph representation)
    graph = color.rgb2gray(img)

    # Apply Normalized Cut for image segmentation
    labels = segmentation.slic(img, compactness=30, n_segments=400)
    segmented_image = color.label2rgb(labels, img, kind="avg")

    titles = [
        "Segmented Image",
    ]

    segmented_image = [segmented_image]

    for j in range(len(titles)):
        # Display the image
        axes[i, j + 1].imshow(segmented_image[j])
        axes[i, j + 1].set_title(f"{titles[j]}")
        axes[i, j + 1].axis("off")

# Show the plot
plt.tight_layout()
plt.show()
