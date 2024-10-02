# Digit Image Classifier (Convolution Neural Network)
Train & test the classic CNN that classifies tiny digit images using MNIST digits dataset. It's often called the "hello world" of deep learning.

Code is in *mnist_digit_classifier.ipynb* (run in Google Colab).

## How to use Git LFS for large model file (optional, in future)
- Initialized Git LFS: `git lfs install`
- `git lfs track "*.keras" "*.onnx" "*.pt"` to track model files with Git LFS, this created `.gitattributes` file.
- Push: `git lfs push origin main`

