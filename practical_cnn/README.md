# CNN with tensorflow
Below structured exercises suggested by ChatGPT:

- [-] basic sequential CNN model with MNIST data (handwritten digits):
    - [x] train & test
    - [-] visualize
    - [ ] try modifying filters, stride, padding etc. to see impact

    NOTE: weird issues with `Sequential` models like *no defined input because model has never been called* - just stick to functional model to avoid all that nonsense.

2. **Convolutions and Filters**:
   - Exercise: Implement a custom convolution operation using `tf.nn.conv2d` to understand kernel operations.
   - Exercise: Visualize feature maps (filters) after the first convolutional layer for an image.

3. **Pooling Layers**:
   - Exercise: Compare MaxPooling and AveragePooling by implementing both in a simple CNN.
   - Challenge: Build your own pooling function using TensorFlow's low-level ops.

4. **Activation Functions**:
   - Exercise: Swap activation functions (`ReLU`, `LeakyReLU`, `Sigmoid`) in the CNN and analyze training curves and output.

5. **Batch Normalization**:
   - Exercise: Add BatchNorm to your CNN model and visualize its effect on training stability.

---

### **Phase 2: Intermediate Topics**
#### **Goal**: Dive deeper into CNN functionalities and practical challenges.

6. **Model Regularization**:
   - Exercise: Apply Dropout and visualize its effect on overfitting.
   - Challenge: Implement L2 regularization on your CNN.

7. **Transfer Learning**:
   - Exercise: Load a pre-trained CNN (e.g., ResNet50) using TensorFlow Hub or `tf.keras.applications`.
   - Modify the model for a new dataset and fine-tune the final layers.

8. **Data Augmentation**:
   - Exercise: Implement data augmentation using `tf.image` and `tf.keras.preprocessing.image`.
   - Challenge: Create a custom data pipeline with `tf.data`.

9. **Custom Layers**:
   - Exercise: Write a custom CNN layer in TensorFlow.
   - Example: Build a learnable attention layer and insert it between two convolutional layers.

10. **Model Optimization**:
    - Exercise: Implement gradient clipping during training.
    - Challenge: Modify the optimizer to include learning rate scheduling.

---

### **Phase 3: Advanced Topics**
#### **Goal**: Work on cutting-edge techniques and concepts in CNNs.

11. **Model Pruning**:
   - Exercise: Perform channel pruning on a CNN, starting with out-channel pruning as you described.
   - Challenge: Write functions to prune CNN layers dynamically based on criteria (e.g., weight magnitudes).

12. **Quantization**:
   - Exercise: Quantize a CNN model using TensorFlow's quantization-aware training APIs.

13. **Branching Architectures**:
   - Exercise: Implement a CNN with skip connections (ResNet-style).
   - Challenge: Add branching (multi-output) to a CNN to solve two tasks simultaneously (e.g., classification and regression).

14. **Explainability**:
   - Exercise: Visualize saliency maps using Grad-CAM.
   - Challenge: Create your own explainability tool by visualizing gradient flows through CNN layers.

15. **Efficient CNN Architectures**:
   - Exercise: Implement depthwise separable convolutions (MobileNet-style).
   - Challenge: Optimize a CNN for deployment on edge devices.

---

### **Phase 4: Full Projects**
#### **Goal**: Consolidate knowledge by applying it to end-to-end projects.

16. **Custom Dataset Training**:
    - Build a CNN from scratch and train it on a custom dataset (e.g., image classification for flowers or fashion).

17. **Image Segmentation**:
    - Exercise: Implement a U-Net style architecture for semantic segmentation.

18. **Object Detection**:
    - Challenge: Use TensorFlow's Object Detection API to fine-tune a model on a custom dataset.

19. **Style Transfer**:
    - Exercise: Implement neural style transfer using pre-trained CNNs.

20. **Generative Models**:
    - Challenge: Build and train a DCGAN for image generation.

---

## **Resources for Practical Exercises**
1. **TensorFlow Documentation**: Official tutorials and guides are structured for hands-on learning.
   - [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
2. **Kaggle Notebooks**: Explore practical implementations and datasets.
   - Search for CNN-related competitions and kernels.
3. **Books and Courses**:
   - _Deep Learning with Python_ by François Chollet (hands-on with Keras/TensorFlow).
   - TensorFlow in Practice Specialization (Coursera).
4. **GitHub Repositories**:
   - Find structured CNN projects and replicate them.
   - Example: Search for CNN pruning or TensorFlow implementations.

---

TODO:
- tensorflow object detection api
- Include explainability (e.g., Grad-CAM visualization)
- transfer learning with pretrained models like ResNet or EfficientNet.
- "fine tuning" and "full fine tuning"
- neural style transfer: https://keras.io/examples/generative/neural_style_transfer/ (keep core of "base" image but with "style" of different image)
- deployment on mobile/edge devices with tensorflow lite
- time series forecasting with 1D CNN - maybe stock market prices; can also Analyze personal fitness data from wearables.

PROBLEMS TO SOLVE:

THIS ONE COULD REALLY BE USEFUL: !
- Detect and organize memes, screenshots, or WhatsApp forwards (which are important, which aren't).
  purpose: add "clear old memes" that deletes only useless stuff like memes, good mornings, not important pics (documents, cards, personal pics).
  obviously be conservative to avoid deleting important stuff like ticket pics.
  IMPL: Train a CNN to detect meme-like features (e.g., text overlaid on images, common watermark styles). Sort screenshots into actionable categories (e.g., tickets, shopping lists, or chat screenshots).

- Google Photos often lumps document photos into "miscellaneous" or "things". (does it, verify, this was said by chatgpt). 
  so can try to train & identify specific docs: govt identity (aadhar, pan), school certificates, electricity bills, etc.
- photo organizer that builds a family tree.  
    - Problem: Google Photos lacks relationship mapping across images.  
    - Solution: Use face recognition to group photos by relationships; allow manual inputs and visual family tree creation.  
    - Features: Add context (e.g., "Uncle at Diwali 2024") and event tags (e.g., "Graduation").  