# ML, Deep Learning & Statistics - Notes & Practice Exercises
Practice Notes on Machine Learning (with code &amp; data for practice exercises). Prefer the most popular libraries - `sklearn` for ML, `pytorch` for Deep Learning.

Sometimes a combination of Deep Learning and ML techniques can be useful. For example, first a Convolutional Neural Network pre-trained on generic images (eg. ImageNet database) is used to identify image features. Then these high-level features are input to ML algo like Support Vector Machine for specific task - eg. whether a tumour is malignant or not. 


## Resources
**Books:**
- [ ] WIP: [Deep Learning: A Visual Approach](DeepLearningVisualApproach/)
- [ ] WIP: [Introduction to Statistical Learning with Python](IntroToStatisticalLearning/)
- [ ] [Statistical Thinking in 21st Century (with Python)](https://statsthinking21.github.io/statsthinking21-python/)
- [ ] Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
- [ ] [Flexible Imputation of Missing Data](https://stefvanbuuren.name/fimd)

**Courses:**
- [ ] WIP: [Statistics & Probability](StanfordStatistics/): WIP: Stanford course @ Coursera

**Videos:**
- NLP with Deep Learning: Youtube Video Playlists of Stanford lectures:
    - [Stanford CS224N](https://youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4)
    - https://youtube.com/playlist?list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp

Implementing ML & Neural Networks (NN) from scratch:
- [ ] WIP: [Impl Deep Learning NN classifier from scratch in APL](neural_network.apl): WIP
- [ ] WIP: [Impl Linear Regression in APL](regression.apl)

## TODO
- [ ] WIP:n Try Classification with [Iris Dataset](https://www.kaggle.com/datasets/uciml/iris): available in `sklearn.datasets.load_iris()`. **WIP:** Did Principal Component Analysis to reduce feature dimensions.
- [ ] Explore ML methods with [tabular datasets](https://dagshub.com/datasets/tabular/).
- [ ] Try making adverserial inputs for famous CNNs like VGG.

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