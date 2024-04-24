import numpy as np
import cv2

cap = cv2.VideoCapture("CV Vision/videos4.mp4")

# Periksa apakah video terbuka dengan benar
if not cap.isOpened():
    print("Gagal membuka video")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        # Video selesai, keluar dari loop
        break

    fgmask = fgbg.apply(frame)

    cv2.imshow("fgmask", fgmask)
    cv2.imshow("frame", frame)

    # Tambahkan penanganan tombol keyboard
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        # Tekan tombol ESC untuk keluar
        break

# Setelah selesai, lepaskan sumber video dan tutup jendela tampilan
cap.release()
cv2.destroyAllWindows()
