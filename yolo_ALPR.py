from cv2 import cv2
import numpy as np
import time

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 512  #608     #Width of network's input image
inpHeight = 512 #608     #Height of network's input image

class_names = ['LP']
colors = []

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolo_ALPR/darknet-yolov3.cfg"
modelWeights = "yolo_ALPR/lapi.weights"


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


class Predictor:
    def __init__(self, silent, backend=cv2.dnn.DNN_BACKEND_DEFAULT, target=cv2.dnn.DNN_TARGET_CPU):
        self.__name__ = "ALPR"
        global colors
        self.silent = silent
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)
        colors = np.random.uniform(0, 255, size=(len(class_names), 3))
    

    def __del__(self):
        cv2.destroyAllWindows()
    

    def set_silent_mode(self, silent):
        self.silent = silent
    

    def detect_in_image(self, img, output_file=None):
        font = cv2.FONT_HERSHEY_SIMPLEX
        assert type(img) == np.ndarray, "Predictor.detect_in_image(self, img): img must be an ndarray"
        height, width, _channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.003921, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        self.net.setInput(blob)
        classes = []
        scores = []
        boxes = []

        t1 = time.time()
        outs = self.net.forward(getOutputsNames(self.net))
        
        for out in outs:
            for detection in out:
                confidences = detection[5:]
                class_id = np.argmax(confidences)
                confidence = confidences[class_id]
                if confidence > confThreshold:
                    # Object detected
                    center_x = detection[0]
                    center_y = detection[1]
                    w = detection[2]
                    h = detection[3]
                    # Rectangle coordinates
                    x = center_x - w / 2
                    y = center_y - h / 2
                    boxes.append(np.array([x, y, x + w, y + h]))
                    scores.append(float(confidence))
                    classes.append(class_id)
        t2 = time.time()
        times = [t2 - t1]
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold)

        for i in indices:
            i = i[0]
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
            cv2.imshow("Output image ALPR", img)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
            cv2.destroyWindow("Output image ALPR")
        
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

            blob = cv2.dnn.blobFromImage(img, 0.003921, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
            self.net.setInput(blob)
            classes = []
            scores = []
            boxes = []

            t1 = time.time()
            outs = self.net.forward(getOutputsNames(self.net))
            
            for out in outs:
                for detection in out:
                    confidences = detection[5:]
                    class_id = np.argmax(confidences)
                    confidence = confidences[class_id]
                    if confidence > confThreshold:
                        # Object detected
                        center_x = detection[0]
                        center_y = detection[1]
                        w = detection[2]
                        h = detection[3]
                        # Rectangle coordinates
                        x = center_x - w / 2
                        y = center_y - h / 2
                        boxes.append(np.array([x, y, x + w, y + h]))
                        scores.append(float(confidence))
                        classes.append(class_id)
            t2 = time.time()
            times.append(t2-t1)
            times = times[-20:]

            vid_boxes.append(boxes)
            vid_scores.append(scores)
            vid_classes.append([float(i) for i in classes])
            vid_times.append(times[-1])            
            
            indices = cv2.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold)

            for i in indices:
                i = i[0]
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
                cv2.imshow("ALPR output", img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                if output_file != None:
                    output.release()
                cv2.destroyAllWindows()
                break
        
        return vid_boxes, vid_scores, vid_classes, vid_times