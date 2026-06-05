import cv2
import os

#using retinaFace model here
#retinaFace gives 5 facial landmarks (left eye, right eye, nose, left mouth corner, right mouth corner) which can be used for better alignment and cropping of faces.
#loading the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

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
            img = cv2.imread(input_path)
            if img is None:
                print(f"Could not read {input_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                print(f"No face detected: {input_path}")
                continue

            # Select largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            #Crop face
            cropped_face = img[y:y+h, x:x+w]

            #saving the cropped face
            save_path = os.path.join(output_dir, file)
            cv2.imwrite(save_path, cropped_face)
            print(f"Saved cropped face: {file}")

print("Face detection and cropping completed.")
    


