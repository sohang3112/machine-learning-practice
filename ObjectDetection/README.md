[2-stage vs 1-stage Object Detectors](https://viso.ai/deep-learning/object-detection/#one-stage-vs-two-stage-deep-learning-object-detection):
- 2-stage ones (eg. RCNN) (typically most accurate but also slower) first identify approximate regions of interest, then crop and classify.
    - R-CNN = Region-based Convolutional Neural Networks : divides image into ~2000 regions, runs CNN on each independently
    - Fast R-CNN speeds up by running CNN once on whole image
- 1-stage ones (eg. YOLO, SSD, RetinaNet) do it in one go: image -> object (labels, bounding boxes)
    - YOLO = You Only Look Once: fastest & most accurate
    - SSD = Single-Shot Detector

Object Recognition is simpler than Object Detection - Recognition doesn't return location info (bounding box).

Evaluation Metric: Mean Average Precision (MAP) (also some others)

Object Detection Components:
- **Backbone** (same as classification): Image -> Features (CNN)
- **Neck** (aggregate features from multiple scales; helps detect objects over wide range of styles) -> 
  **Head** (outputs final: bounding boxes, class labels of each object): Features -> Detection

Key features of any object detection model:
- Backbone: which used?
- Head/Neck/Backbone attachments
- Anchor Boxes

smaller objects detected earlier in backbone CNN, larger objects in later layers

YOLO table - [source](https://www.coursera.org/learn/deep-learning-object-detection/supplement/NEFwE/yolo-detectors-in-matlab-reference)

| Framework | Backbone Options    | Neck Attachments | Anchor Boxes
| --------- | ------------------- | ---------------- | ------------
| YOLO v4   | csp-darknet-53-coco | 3 | yes - need a multiple of 3
| _         |  tiny-yolov4-coco   | 2 | yes - need a multiple of 2
| YOLOX     | small-coco          | 3 | no
| _         | tiny-coco           | 3 | no