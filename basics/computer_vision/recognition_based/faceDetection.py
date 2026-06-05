import cv2
import os

INPUT_DIR = "dataset_fiveFaces"
OUTPUT_DIR = "cropped_faces"
MODEL_PATH="face_detection_yunet_2023mar.onnx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#Loading YuNet
detector=cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "", 
    (320,320),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000
)

#Processing images
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    input_path=os.path.join(INPUT_DIR, filename)
    img=cv2.imread(input_path)
    if img is None:
        print(f"Could not read {filename}")
        continue
    h,w=img.shape[:2]

    #giving image size to YuNet
    detector.setInputSize((w,h))
    _,faces=detector.detect(img)

    if faces is None or len(faces)==0:
        print(f"No faces detected in {filename}")
        continue

    #selecting highest confidence face
    best_face=max(faces,key=lambda f:f[-1])
    x,y,fw,fh=best_face[:4]
    x=max(0,int (x))
    y=max(0,int(y))
    fw=int (fw)
    fh=int (fh)

    #Crop face
    face_crop=img[y:y+fh,x:x+fw]
    output_path=os.path.join(OUTPUT_DIR,filename)
    cv2.imwrite(output_path,face_crop)
    print(f"Saved: {filename}")

print("\nDONE")