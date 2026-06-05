import cv2
import os
from mtcnn import MTCNN

detector = MTCNN()

#input and output folder
input_root = "dataset_fiveFaces"
output_root = "cropped_faces"

os.makedirs(output_root, exist_ok=True)

for root, dirs, files in os.walk(input_root):

    #Relative path from input dataset
    relative_path=os.path.relpath(root, input_root)

    #creating same directory structures
    output_dir=os.path.join(output_root, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    for file in files:

        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(root, file)
            img_bgr = cv2.imread(input_path)
            if img_bgr is None:
                print(f"Could not read {input_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)

            if len(faces) == 0:
                print(f"No face detected: {input_path}")
                continue

            # Select best face
            best_face = max(faces, key=lambda face:face['confidence'])
            x, y, w, h = best_face['box']

            x=max(0, x)
            y=max(0, y)

            #small padding
            pad=int(0.001*max(w,h))
            x1=max(0, x-pad)
            y1=max(0, y-pad)
            x2=min(img_rgb.shape[1], x+w+pad)
            y2=min(img_rgb.shape[0], y+h+pad)
            crop=img_bgr[y1:y2, x1:x2]
            crop=cv2.resize(crop, (160, 160))
            
            #saving the cropped face
            save_path=os.path.join(output_dir, file)
            cv2.imwrite(save_path, crop)
            print("Saved:", save_path)
          
print("Face detection and cropping completed.")
    


