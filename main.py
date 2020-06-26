import csv
import numpy as np
from cv2 import cv2
from yolo_tf import Predictor

def yolo_tf():
    #detect_in_video(output_name=None)
    pass

def write_csv(boxes, scores, classes, times):
    with open("test.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for i in range(len(boxes)):
            for j in range(len(classes[i])):
                writer.writerow((i, boxes[i][j], classes[i][j], scores[i][j], times[i]))


if __name__ == "__main__":
    #yolo_tf()
    pred = Predictor()
    #print(pred.detect_in_image(cv2.imread("test.jpg")))
    cap = cv2.VideoCapture("videos/test.avi")
    boxes, scores, classes, times = pred.detect_in_video(cap, "output/test_YOLO_tf.avi")
    cap.release()
    write_csv(boxes, scores, classes, times)