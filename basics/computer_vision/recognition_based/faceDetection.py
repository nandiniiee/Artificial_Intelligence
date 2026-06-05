import cv2
import os

INPUT_DIR = "dataset_fiveFaces"
OUTPUT_DIR = "cropped_faces"
MODEL_PATH = "face_detection_yunet_2023mar.onnx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loading YuNet
detector = cv2.FaceDetectorYN.create(
    MODEL_PATH, "", (320, 320), score_threshold=0.8, nms_threshold=0.3, top_k=5000
)

# Processing images
for root, dirs, files in os.walk(INPUT_DIR):
    for filename in files:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        input_path = os.path.join(root, filename)
        img = cv2.imread(input_path)

        if img is None:
            print(f"Could not read {input_path}")
            continue
        h, w = img.shape[:2]

        # YuNet requires input size to be set before detection
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        if faces is None or len(faces) == 0:
            print(f"No face detected: {input_path}")
            continue

        # finding best face based on confidence score
        best_face = max(faces, key=lambda f: f[-1])
        x, y, fw, fh = best_face[:4]

        x = max(0, int(x))
        y = max(0, int(y))
        fw = int(fw)
        fh = int(fh)

        # Cropping the face
        face_crop = img[y : y + fh, x : x + fw]
        relative_path = os.path.relpath(root, INPUT_DIR)
        output_subdir = os.path.join(OUTPUT_DIR, relative_path)
        # saving the cropped face
        os.makedirs(output_subdir, exist_ok=True)
        output_path = os.path.join(output_subdir, filename)
        cv2.imwrite(output_path, face_crop)
        print(f"Saved: {filename}")

print("\nDONE")
