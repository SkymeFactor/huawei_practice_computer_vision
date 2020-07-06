import csv
import numpy as np
import argparse as ap
import os
from cv2 import cv2
from yolo_tf import Predictor as yolo_pred
from mobileSSD import Predictor as ssd_pred
from yolo_ALPR import Predictor as alpr_pred


kernels = [(1, 1), (3, 3), (5, 5), (9, 9)]
sigmas = [1, 5, 15, 50]
means = [1, 5, 15, 50]
stddevs = [1, 5, 15, 50]


def write_csv(filename, boxes, scores, classes, times):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for i in range(len(boxes)):
            for j in range(len(classes[i])):
                writer.writerow((i, boxes[i][j], classes[i][j], scores[i][j], times[i]))


def apply_gaussian_blur(cap, output_file, kernel, sigma, silent=True):
    _, img = cap.read()
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    output = cv2.VideoWriter(output_file, fourcc, 30.0, (img.shape[1], img.shape[0]))
    while True:
        if type(img) == type(None):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
        img = cv2.GaussianBlur(img, kernel, sigma)
        if silent == False:
            cv2.imshow("Output_blur", img)
        output.write(img)
        _, img = cap.read()
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
    if silent == False:
        cv2.destroyWindow("Output_blur")
    output.release()


def apply_noise(cap, output_file, mean, stddev, silent=True):
    _, img = cap.read()
    
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    output = cv2.VideoWriter(output_file, fourcc, 30.0, (img.shape[1], img.shape[0]))
    while True:
        if type(img) == type(None):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
        dst = np.empty_like(img)
        noise = cv2.randn(dst, mean, stddev)
        img = cv2.add(img, noise)
        if silent == False:
            cv2.imshow("Output_noise", img)
        output.write(img)
        _, img = cap.read()
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
    if silent == False:
        cv2.destroyWindow("Output_noise")
    output.release()


class Predictors:
    def __init__(self, *preds):
        self.preds = []
        for p in preds:
            self.preds.append(p)
    
    def obtain_filename(self, source):        
        return (args.video.split("/"))[-1].split(".")[0]

    def iterate_through_all(self, source):
        cap = cv2.VideoCapture(source)
        filename = self.obtain_filename(source)
        try:
            os.mkdir("output/" + filename)
        except Exception:
            pass
            
        # Hardcode alert, has to be fixed!!!
        for ker in kernels:
            for sig in sigmas:
                for mn in means:
                    for stdd in stddevs:
                        apply_gaussian_blur(cap, "videos/_" + filename + "_temp.avi",ker, sig)
                        temp = cv2.VideoCapture("videos/_" + filename + "_temp.avi")
                        apply_noise(temp, "videos/__" + filename + "_temp.avi", mn, stdd)
                        temp.release()
                        temp = cv2.VideoCapture("videos/__" + filename + "_temp.avi")
                        for pred in self.preds:
                            boxes, scores, classes, times = pred.detect_in_video(temp)
                            write_csv("output/"+filename+"/"+str(pred.__name__)+"_"+filename+"_"+str(ker)+"_"+str(sig)+"_"+str(mn)+"_"+str(stdd)+
                                ".csv", boxes, scores, classes, times)
                            temp.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        temp.release()
        
        cap.release()
        os.remove("videos/__"+filename+"_temp.avi")
        os.remove("videos/_"+filename+"_temp.avi")


if __name__ == "__main__":
    # Parse the arguments
    parser = ap.ArgumentParser(
        description='Script to run the video file on all available networks and gather their worktime statistics')
    parser.add_argument("--video", default="videos/test.avi", help="Path to video file")
    parser.add_argument("--silent", default=True, type=lambda silent: False if (silent == "False" or silent == "false") else True,
        help="True by default, False to set visible mode")
    args = parser.parse_args()
    
    
    # Create a bunch of predictors from different networks to iterate through them
    pred = Predictors(yolo_pred(silent=args.silent),
        ssd_pred(silent=args.silent, backend=cv2.dnn.DNN_BACKEND_VKCOM, target=cv2.dnn.DNN_TARGET_VULKAN),
        alpr_pred(silent=args.silent, backend=cv2.dnn.DNN_BACKEND_DEFAULT, target=cv2.dnn.DNN_TARGET_CPU) )
    # Perform the iteration process with video file
    pred.iterate_through_all(args.video)

    