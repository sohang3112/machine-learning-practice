# ML, Deep Learning & Statistics - Notes & Practice Exercises
Practice Notes on Machine Learning (with code &amp; data for practice exercises). Prefer the most popular libraries - `sklearn` for ML, `pytorch` for Deep Learning.

Sometimes a combination of Deep Learning and ML techniques can be useful. For example, first a Convolutional Neural Network pre-trained on generic images (eg. ImageNet database) is used to identify image features. Then these high-level features are input to ML algo like Support Vector Machine for specific task - eg. whether a tumour is malignant or not. 


## Resources
In addition to the below, there are also many useful courses in 
[Datacamp](https://app.datacamp.com/learn/courses).

**Books:**
- [ ] WIP: [Deep Learning: A Visual Approach](DeepLearningVisualApproach/)
- [ ] Fast.ai Practical Deep Learning for Coders: [Course](https://course.fast.ai/)and [Book](https://course.fast.ai/Resources/book.html) - uses PyTorch
- [ ] WIP: [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow)
- [ ] WIP: [Introduction to Statistical Learning with Python](IntroToStatisticalLearning/)
    - This book's contents are also available in video format in [a YouTube playlist](https://www.youtube.com/playlist?list=PLOg0ngHtcqbPTlZzRHA2ocQZqB1D_qZ5V).
- [ ] Applied Predictive Modeling
- [ ] [Statistical Thinking in 21st Century (with Python)](https://statsthinking21.github.io/statsthinking21-python/)
- [ ] [Flexible Imputation of Missing Data](https://stefvanbuuren.name/fimd)

**Courses:**
- [x] [Statistics & Probability](StanfordStatistics/): Stanford course @ Coursera
- [ ] [Machine Learning by Andrew Ng](https://www.coursera.org/specializations/machine-learning-introduction): 3-course series @ Coursera

**Blogs:**
- [Statistics by Jim](https://statisticsbyjim.com/)

**Videos:**
- NLP with Deep Learning: Youtube Video Playlists of Stanford lectures:
    - [Stanford CS224N](https://youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4)
    - https://youtube.com/playlist?list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp

### Computer Vision
**Courses:**
- [ ] [OpenCV Course](https://courses.opencv.org/courses/course-v1:OpenCV+Bootcamp+CV0/course/)
- [ ] [Computer Vision with Embedded ML](https://www.coursera.org/learn/computer-vision-with-embedded-machine-learning) - Coursera


## TODO
- Read about Regression. Covered Linear Regression in [Statistics & Probability](StanfordStatistics/) course, now study Logistic Regression, and how to modify ML & Deep Learning methods for regression.
- [ ] WIP: Try Classification with [Iris Dataset](https://www.kaggle.com/datasets/uciml/iris): available in `sklearn.datasets.load_iris()`. **WIP:** Did Principal Component Analysis to reduce feature dimensions.
- [ ] Explore ML methods with [tabular datasets](https://dagshub.com/datasets/tabular/).
- [ ] Try making adverserial inputs for famous CNNs like VGG.

### Projects
- [ ] WIP: [Digits Classifier](digit-classifier/)

### Kaggle Problems to Solve
- [ ] [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk): Probabilistic Classification
- [ ] [Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction): Probabilistic Classification
- [-] [VSB Power Line Fault Detection](https://www.kaggle.com/c/vsb-power-line-fault-detection): Binary Classification: **WIP** (trying it in Kaggle Notebook)
- [ ] [Microsoft Malware Prediction](https://www.kaggle.com/c/microsoft-malware-prediction): Binary Classification
- [ ] [IEEE Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection): Binary Classification


## Landmark Research Papers to read

### Machine Learning
- Lasso method (1996) - var selection & regularization to imp accuracy & interpretability
- AdaBoost (1997)

### Deep Learning
- *Gradient-Based Learning Applied to Document Recognition (1998)* - introduced CNN architecture
- AlexNet (2012) - won ImageNet classification competition using Deep CNN
- *Generative Adverserial Nets (2014)* - introduced GAN architecture
- *Sequence to Sequence Learning with Neural Networks (2014)* - introduced Seq2Seq
- *Attention is all you need (2017)* - introduced Transformers architecure
- BeRT (2018) - NLP tasks using Transformer architecture
- AlphaGo (2016) - combined Deep Reinforcement Learning with Monte Carlo tree search
- Neural Architecural Search (2018) - automated method for designing neural networks