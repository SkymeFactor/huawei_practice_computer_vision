from cv2 import cv2
import numpy as np
import time
import argparse

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network')
parser.add_argument("--prototxt", default="mobileSSD/MobileNetSSD_deploy.prototxt")
parser.add_argument("--weights", default="mobileSSD/MobileNetSSD_deploy.caffemodel")
parser.add_argument("--thr", default=0.3, type=float)
args = parser.parse_args()

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
# Labels of Network.
classes = [ 'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor' ]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Swith to Nvidia GPU if opencv is compiled with CUDA backend
# Might be changed to OpenCL if necessarry, it is much slower though
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load the video capture <<<--------------------------------------------------------
cap = cv2.VideoCapture("videos/test.avi")
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
output = cv2.VideoWriter("output/test_SSD.avi", fourcc, 24.0, (1280, 720))
#img = cv2.imread("test.jpg")
#img = cv2.resize(img, None, fx=0.8, fy=0.8)
font = cv2.FONT_HERSHEY_SIMPLEX
start_time = time.time()
frame_id = 0

while True:
    _, img = cap.read()
    frame_id += 1
    if type(img) == np.ndarray:
        height, width, channels = img.shape
    else:
        break

    blob = cv2.dnn.blobFromImage(img, 0.007843, (512, 512), (127.5, 127.5, 127.5), crop=False)
    net.setInput(blob)
    detections = net.forward()

    class_ids = []
    confidences = []
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        if confidence > args.thr:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            box -= [0, 0, box[0], box[1]]
            boxes.append(box.astype("int"))
            confidences.append(confidence)
            class_ids.append(class_id)
        
    #indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    for i in range(len(boxes)):
        #if i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]]) + " " + str(round(confidences[i], 2))
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x + 5, y + 25), font, 1, color, 2)
    elapsed_time = time.time() - start_time
    fps = frame_id / elapsed_time
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 1, (255, 255, 255), 3)
    img = cv2.resize(img, (1280, 720))
    output.write(img)
    
    # Uncomment in order to see the output on your screen,
    # otherwise it will be running in silent mode
    #cv2.imshow("Output result", img) # <-----------------

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
output.release()
cv2.destroyAllWindows()
