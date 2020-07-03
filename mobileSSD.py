from cv2 import cv2
import numpy as np
import time

#Load the Caffe model
prototxt = "mobileSSD/MobileNetSSD_deploy.prototxt"
weights = "mobileSSD/MobileNetSSD_deploy.caffemodel"


# Labels of Network.
class_names = [ 'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor' ]
colors = []


class Predictor:
    def __init__(self, silent=False, threshold=0.3, backend=cv2.dnn.DNN_BACKEND_DEFAULT, target=cv2.dnn.DNN_TARGET_CPU):
        self.__name__ = "SSD"
        global colors
        self.silent = silent
        self.net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)
        self.threshold = threshold
        colors = np.random.uniform(0, 255, size=(len(class_names), 3))
    

    def __del__(self):
        cv2.destroyAllWindows()
    

    def set_silent_mode(self, silent):
        self.silent = silent
    

    def detect_in_image(self, img, output_file=None):
        font = cv2.FONT_HERSHEY_SIMPLEX
        assert type(img) == np.ndarray, "Predictor.detect_in_image(self, img): img must be an ndarray"
        height, width, _channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.007843, (512, 512), (127.5, 127.5, 127.5), 1, crop=False)
        self.net.setInput(blob)
        classes = []
        scores = []
        boxes = []

        t1 = time.time()
        # Making the prediction of a frame
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            if confidence > self.threshold:
                box = detections[0, 0, i, 3:7]
                boxes.append(box)
                scores.append(confidence)
                classes.append(class_id)
        t2 = time.time()
        times = [t2 - t1]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = (boxes[i] * np.array([width, height, width, height])).astype("int")
            label = str(class_names[classes[i]]) + " " + str(round(scores[i], 2))
            color = colors[classes[i]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1 + 5, y1 + 25), font, 1, color, 2)
        cv2.putText(img, "Time: " + str(round(sum(times)/len(times)*1000, 2)) +" ms", (10, 30),
                        font, 1, (0, 0, 0), 4)
        cv2.putText(img, "Time: " + str(round(sum(times)/len(times)*1000, 2)) +" ms", (10, 30),
                        font, 1, (255, 255, 255), 2)
        img = cv2.resize(img, (1280, 720))

        if self.silent == False:
            cv2.imshow("Output image SSD", img)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
            cv2.destroyWindow("Output image SSD")
        
        if output_file != None:
            cv2.imwrite(output_file, img)

        return [boxes], [scores], [classes], times
    

    def detect_in_video(self, cap, output_file = None):
        if output_file != None:
            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            output = cv2.VideoWriter(output_file, fourcc, 30.0, (1280, 720))

        font = cv2.FONT_HERSHEY_SIMPLEX
        start_time = time.time()
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_id = 0
        vid_boxes, vid_scores, vid_classes, vid_times = [], [], [], []
        times = []

        while True:
            frame_id += 1
            _, img = cap.read()
            if type(img) == np.ndarray:
                height, width, _channels = img.shape
            else: break

            blob = cv2.dnn.blobFromImage(img, 0.007843, (512, 512), (127.5, 127.5, 127.5), 1, crop=False)
            self.net.setInput(blob)
            classes = []
            scores = []
            boxes = []
            
            t1 = time.time()
            # Making the prediction of a frame
            detections = self.net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])
                if confidence > self.threshold:
                    box = detections[0, 0, i, 3:7]
                    boxes.append(box)
                    scores.append(confidence)
                    classes.append(class_id)
            t2 = time.time()
            times.append(t2-t1)
            times = times[-20:]

            vid_boxes.append(boxes)
            vid_scores.append(scores)
            vid_classes.append([float(i) for i in classes])
            vid_times.append(times[-1])
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = (boxes[i] * np.array([width, height, width, height])).astype("int")
                label = str(class_names[classes[i]]) + " " + str(round(scores[i], 2))
                color = colors[classes[i]]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1 + 5, y1 + 25), font, 1, color, 2)
            elapsed_time = time.time() - start_time
            fps = frame_id / elapsed_time
            # Draw the text twice in order to make an outline.
            # FPS counter
            cv2.putText(img, "FPS: " + str(round(fps, 2)),
                        (10, 100), font, 1, (0, 0, 0), 4)
            cv2.putText(img, "FPS: " + str(round(fps, 2)),
                        (10, 100), font, 1, (255, 255, 255), 2)
            # Time per frame in ms
            cv2.putText(img, "Time: " + str(round(sum(times)/len(times)*1000, 2)) +" ms", (10, 30),
                            font, 1, (0, 0, 0), 4)
            cv2.putText(img, "Time: " + str(round(sum(times)/len(times)*1000, 2)) +" ms", (10, 30),
                            font, 1, (255, 255, 255), 2)
            img = cv2.resize(img, (1280, 720))
            
            if output_file != None:
                output.write(img)
            
            if self.silent == False:
                cv2.imshow("SSD output", img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                if output_file != None:
                    output.release()
                cv2.destroyAllWindows()
                break
        
        cv2.destroyWindow("SSD output")

        return vid_boxes, vid_scores, vid_classes, vid_times