According to ChatGPT (haven't verified), these are the layers of YOLOv3 Tiny Image Classifier CNN:
```
Input
  |
Conv (16 filters, 3x3, padding 1) + BN + Leaky ReLU
  |
MaxPool (2x2, stride 2)
  |
Conv (32 filters, 3x3, padding 1) + BN + Leaky ReLU
  |
MaxPool (2x2, stride 2)
  |
Conv (64 filters, 3x3, padding 1) + BN + Leaky ReLU
  |
MaxPool (2x2, stride 2)
  |
Conv (128 filters, 3x3, padding 1) + BN + Leaky ReLU
  |
MaxPool (2x2, stride 2)
  |
Conv (256 filters, 3x3, padding 1) + BN + Leaky ReLU
  |
MaxPool (2x2, stride 2)
  |
Conv (512 filters, 3x3, padding 1) + BN + Leaky ReLU
  |
MaxPool (2x2)
  |
Conv (1024 filters, 3x3, padding 1) + BN + Leaky ReLU
  |
Conv (256 filters, 1x1, padding 0) + BN + Leaky ReLU
  |
Conv (512 filters, 3x3, padding 1) + BN + Leaky ReLU
  |
Conv (255 filters, 1x1, padding 0) (Detection Layer 1)
  |
Conv (128 filters, 1x1, padding 0) + BN + Leaky ReLU
  |
UpSample (scale 2x)
  |
Concatenate (feature map from layer before Detection Layer 1)
  |
Conv (256 filters, 3x3, padding 1) + BN + Leaky ReLU
  |
Conv (255 filters, 1x1, padding 0) (Detection Layer 2)
```

- **Detection Layers** `Conv (255 filters, 1x1, padding 0)` are output layers where the predictions are made. Each detection layer outputs bounding box coordinates, objectness score, and class probabilities. Padding 0 makes sense since 1x1 is footprint size (i.e., processing just 1 cell in each filter). Since it's a 1x1 filter, it reduces no. of channels in output tensor.
- **Fully Convolutional** (i.e., no dense layers) because convolutional layers can handle variable-sized input tensors (images of different dimensions), preserve spatial information (important for tasks like object detection), and are more efficient.

*BN* = Batch Normalization (shift & scale layer outputs to fit in range $[-1,1]$)

### My Questions
- In starting convolution layers:
    - Why both stride & pooling?
    - 