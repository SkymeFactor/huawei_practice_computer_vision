
# Summer practice at Huawei
## Description
Computer vision using `opencv` and fast object detectors e.g. yolo, ssd etc.

The purpose of this practice is to observe how different object detectors behave when different hindrances are being applied such as noise, blur etc. comparing to original video with no effects and gather statistics of their work-process.
#### Statistics in CSV file, structure example:
| frame |  x0  |  y0  |  x1  |  y1  | class | confidence | frame time |
|-------|------|------|------|------|-------|------------|------------|
|  657  | 0.49 | 0.62 | 0.52 | 0.69 |  3.0  |   0.6292   |   0.0501   |
#### Detectors that have been used during this work:
- YOLOv3 on tensorflow-gpu
- mobileSSD on cv2.dnn
- YOLOv3 based ALPR on cv2.dnn

I beware you that some functions were specifically designed to work with NVidia CUDA and have never been tested with other devices, so in order to make it possible you have to change the source code.

## Software dependencies:
- Python 3.x
- Tensorflow-gpu >=2.2
- Nvidia CUDA toolkit >=10.0
- cuDNN >=7.0
- OpenCV >=4.0
- Numpy
- argparse
## Usage:
First of all you have to make sure that you have all requirements installed by running

`pip install -r requirements.txt`

Note, that the installation of python, CUDA and cuDNN is user's responsibility.

Next, make sure that you have the following files:
```
.
├── mobileSSD
│ 	├── MobileNetSSD_deploy.caffemodel
│ 	└── MobileNetSSD_deploy.prototxt
├── yolo_ALPR
│ 	├── classes.names
│ 	├── darknet-yolov3.cfg
│ 	├── lapi.weights
└── yolo_coco
	├── yolov3.cfg
	└── yolov3.weights
```
If you're experiencing some troubles with git LFS and/or somehow cannot get these files you may download them directly by following links:

[mobileSSD](https://github.com/djmv/MobilNet_SSD_opencv)

[yolo_ALPR](https://www.kaggle.com/achrafkhazri/yolo-weights-for-licence-plate-detector)

[yolo_coco/yolov3.cfg](https://github.com/x4nth055/pythoncode-tutorials/blob/master/machine-learning/object-detection/cfg/yolov3.cfg)

[yolo_coco/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

Finally, run the command below with your own video example

`python main.py --video /videos/test.avi --silent=True`

Once the execution ends you'll get the outputs and statistics in the `./output` folder
