import csv
import numpy as np
from cv2 import cv2
from yolo_tf import Predictor

def yolo_tf():
    #detect_in_video(output_name=None)
    pass

def write_csv(filename, boxes, scores, classes, times):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for i in range(len(boxes)):
            for j in range(len(classes[i])):
                writer.writerow((i, boxes[i][j], classes[i][j], scores[i][j], times[i]))

def make_gaussian_blur(source, output_file, kernel, sigmaX, silent=True):
    cap = cv2.VideoCapture(source)
    _, img = cap.read()
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    output = cv2.VideoWriter(output_file, fourcc, 30.0, (img.shape[1], img.shape[0]))
    while True:
        if type(img) == type(None):
            break
        img = cv2.GaussianBlur(img, kernel, sigmaX)
        if silent == False:
            cv2.imshow("Output_blur", img)
        output.write(img)
        _, img = cap.read()
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
    if silent == False:
        cv2.destroyWindow("Output_blur")
    cap.release()
    output.release()

if __name__ == "__main__":
    #yolo_tf()
    #make_gaussian_blur("videos/test.avi", "videos/test_gaussian_blur.avi", (5, 5), 1)
    
    pred = Predictor()

    #print(pred.detect_in_image(cv2.imread("test.jpg")))

    cap = cv2.VideoCapture("videos/test.avi")
    boxes, scores, classes, times = pred.detect_in_video(cap, "output/test_YOLO_tf.avi")
    write_csv("test.csv", boxes, scores, classes, times)
    cap.release()

    cap = cv2.VideoCapture("videos/test_gaussian_blur.avi")
    boxes, scores, classes, times = pred.detect_in_video(cap, "output/test_YOLO_tf_gaussian_blur.avi")
    write_csv("test_gaussian.csv", boxes, scores, classes, times)
    cap.release()
