import os
import cv2
import numpy as np

# folder containing nandini's images
folder = "dataset_sixFaces/nandini"
print("Starting augemntaion\n")
for file in os.listdir(folder):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    # Skipping already augmented images
    if any(
        tag in file
        for tag in [
            "_flip",
            "_rotp10",
            "_rotn10",
            "_bright",
            "_dark",
            "_contrast_high",
            "_contrast_low",
            "_noise",
            "_blur",
            "_jpeg",
            "_shift",
        ]
    ):
        continue
    image_path = os.path.join(folder, file)
    img = cv2.imread(image_path)
    if img is None:
        continue
    filename = os.path.splitext(file)[0]
    h, w = img.shape[:2]

    # HORIZONTAL FLIP
    flip = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(folder, f"{filename}_flip.jpg"), flip)

    # ROTATING +10 DEGREES
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1)
    rot_p10 = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite(os.path.join(folder, f"{filename}_rotp10.jpg"), rot_p10)

    # ROTATING -10 DEGREES
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -10, 1)
    rot_n10 = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite(os.path.join(folder, f"{filename}_rotn10.jpg"), rot_n10)

    # INCREASING BRIGHTNESS
    bright = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    cv2.imwrite(os.path.join(folder, f"{filename}_bright.jpg"), bright)

    # DECREASING BRIGHTNESS
    dark = cv2.convertScaleAbs(img, alpha=1.0, beta=-30)
    cv2.imwrite(os.path.join(folder, f"{filename}_dark.jpg"), dark)

    # CONTRAST INCREASE
    contrast_high = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
    cv2.imwrite(os.path.join(folder, f"{filename}_contrast_high.jpg"), contrast_high)

    # CONTRAST DECREASE
    contrast_low = cv2.convertScaleAbs(img, alpha=0.7, beta=0)
    cv2.imwrite(os.path.join(folder, f"{filename}_contrast_low.jpg"), contrast_low)

    # GAUSSIAN NOISE
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(folder, f"{filename}_noise.jpg"), noisy)

    # GAUSSIAN BLUR
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imwrite(os.path.join(folder, f"{filename}_blur.jpg"), blur)

    # COMPRESSION
    cv2.imwrite(
        os.path.join(folder, f"{filename}_jpeg.jpg"),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 40],
    )

    # TRANSLATION
    M = np.float32([[1, 0, 10], [0, 1, 10]])
    shifted = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite(os.path.join(folder, f"{filename}_shift.jpg"), shifted)
    print(f"Augmented: {file}")

    # ZOOM IN
    zoom_factor = 1.1
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    zoom_in = img[y1 : y1 + new_h, x1 : x1 + new_w]
    zoom_in = cv2.resize(zoom_in, (w, h))
    cv2.imwrite(os.path.join(folder, f"{filename}_zoomin.jpg"), zoom_in)

    # ZOOM OUT
    zoom_factor = 0.9
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)
    small = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros_like(img)
    x_offset = (w - new_w) // 2
    y_offset = (h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = small
    cv2.imwrite(os.path.join(folder, f"{filename}_zoomout.jpg"), canvas)

print("\nAugmentation completed.")
