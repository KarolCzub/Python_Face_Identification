import numpy as np
import cv2
import os

subjects = ["", "Karol Czub"]

#wykrywanie pojedynczej twarzy do nauki
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('D:/materialy/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

#wykrywanie wszystkich twarzy przez kamere
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('D:/materialy/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5);
    return faces


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels


def draw_green_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

def draw_red_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    faces = detect_faces(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for face in faces:
        if not face is None:
            (x, y, w, h) = face
            rect = gray[y:y+w, x:x+h]
            label, confidence = face_recognizer.predict(rect)
            print ("pewnosc wynosi: ", confidence)
            label_text = subjects[label]
            if (confidence > 60 and confidence < 100):
                draw_green_rectangle(img, face)
                draw_text(img, label_text, face[0], face[1] - 5)
            else:
                draw_red_rectangle(img, face)
    return img

print("Preparing data...")
faces, labels = prepare_training_data("D:/materialy/twarze")
print("Data prepared. Total faces detected to train: ", len(faces))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
print("face recognizer trained. Starting...")

cap = cv2.VideoCapture('http://192.168.0.100:8080/video')

while (True):
    ret, frame = cap.read()
    frame = predict(frame)
    cv2.imshow("face recogniser", cv2.resize(frame, (300, 500)))
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()