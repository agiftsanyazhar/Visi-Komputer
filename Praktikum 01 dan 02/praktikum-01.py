import cv2

img = cv2.imread("Praktikum 01 dan 02/img/Logo Unair.png")

print("Ukuran Image: ", img.shape)
print("Type Data: ", img.dtype)

cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
