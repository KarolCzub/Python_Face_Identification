import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
import sys
import json
import numpy as np
import tensorflow as tf


'''
each face is categorized in 3 types: Center, Left, Right 
min percentage threshold is 70% 
'''

def identifyPeople(capture):
    tensorGraph = tf.Graph();
    aligner = AlignCustom();
    allFeatures = FaceFeature(tensorGraph)
    face_detect = MTCNNDetect(tensorGraph, scale_factor=2);
    print("[INFO] camera sensor warming up...")
    while True:
        _, frame = capture.read();
        rects, landmarks = face_detect.detect_face(frame, 80);  # min face size is set to 80x80
        aligns = []
        positions = []
        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160, frame, landmarks[i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else:
                print("Align face failed")
        if (len(aligns) > 0):
            features_arr = allFeatures.get_features(aligns)
            recog_data = getKnownPeople(features_arr, positions);
            for (i, rect) in enumerate(rects):
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
                              (255, 0, 0))  # draw bounding box for the face
                cv2.putText(frame, recog_data[i][0] + " - " + str(recog_data[i][1]) + "%", (rect[0], rect[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


def addNewPerson(capture, newLabel):
    tensorGraph = tf.Graph();
    aligner = AlignCustom();
    allFeatures = FaceFeature(tensorGraph)
    face_detect = MTCNNDetect(tensorGraph, scale_factor=2);
    f = open('./facerec_128D.txt', 'r');
    data_set = json.loads(f.read());
    person_imgs = {"Left": [], "Right": [], "Center": []};
    person_features = {"Left": [], "Right": [], "Center": []};
    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset");
    while True:
        _, frame = capture.read();
        rects, landmarks = face_detect.detect_face(frame, 80);  # min face size is set to 80x80
        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(160, frame, landmarks[i]);
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame)
                cv2.imshow("Captured face", aligned_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    print("Capturing new face has ended");
    for pos in person_imgs:
        person_features[pos] = [np.mean(allFeatures.get_features(person_imgs[pos]), axis=0).tolist()]
    data_set[newLabel] = person_features;
    f = open('./facerec_128D.txt', 'w');
    f.write(json.dumps(data_set))
    print("new face saved");


'''
facerec_128D.txt Data Structure:
{
"Person ID": {
    "Center": [[128D vector]],
    "Left": [[128D vector]],
    "Right": [[128D Vector]]
    }
}
'''


def getKnownPeople(features_arr, positions, thres=0.6, percent_thres=70):
    f = open('./facerec_128D.txt', 'r')
    data_set = json.loads(f.read());
    returnRes = [];
    for (i, features_128D) in enumerate(features_arr):
        result = "Unknown";
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]];
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data - features_128D)))
                if (distance < smallest):
                    smallest = distance;
                    result = person;
        percentage = min(100, 100 * thres / smallest)
        if percentage <= percent_thres:
            result = "Unknown"
        returnRes.append((result, percentage))
    return returnRes



