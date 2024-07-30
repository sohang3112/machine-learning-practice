Notes from reading book **Deep Learning: A Visual Approach**.

## Topics Covered - Revise

### Machine Learning (ML)
- *Statistics & Probability*: Variance (formula using Mean), Probability Distributions (Uniform, Normal) discrete categorical distributions (Bernoulli & Multinoulli), Three Sigma Rule, Interquartile Range, Covariance vs Correlation, Partial Correlation, Bootstrapping, Bayes Posterior-Prior loop
- *Information Theory*: Adaptive (Variable Bitrate) code, Compression Ratio, Entropy(self, cross, relative), KL Divergence
- *Data Preprocessing*: Imputation (of missing data), Categorical (Nominal data, Ordinal vs One-Hot encoding), Slicing (Elem-wise, Feature-wise, Sample-wise), Transforms (Normalize, Standardize, Univariate, Multivariate, Inverse), Augmentation
    - *Data Shrinking*:
        - *Dimensionality Reduction*: PCA (Mean Squared Error, Explainable Variance Ratio, Eigenvectors of covariance matrix)
        - *Feature Selection*: Feature Importance estimation
- *Performance Metrics*:
    - *Regression*: Mean Absolute Error ; **TODO:** Study seperately
    - *Discrete Classification*: Type I,II Errors, Accuracy, Precision, Recall, Specificity, F1 Score, (*optional* Matthews Correlation Coefficient)
    - *Probabilistic Classification*: Area Under ROC Curve ; **TODO:** Study seperately
- *ML Concepts*: IID vars (Independant & Identically Distributed), Test-Train-Validation split, mistakes (Data leakage, Information leakage), k-Fold Cross Validation, Underfitting & Overfitting, Bias-Variance Tradeoff, Regularization $\lambda$
- *ML Model Types*: Supervised, Unsupervised, Semi-Supervised (Autoencoder), Reinforcement Learning (RL), Parametric vs Non-Parametric 
- *ML Basic Models*: SVM, Naive Bayes, kNN, Decision Trees, Multi-Class (One vs One, One vs Rest)
- *ML Ensemble*: Law of Diminishing Returns, Bagging, Random Forests vs Extremely Random Trees
    - *Boosting*: Weighted Plurality, Weak Learner, Decision Stump
        - *Binary Classification*: Adaboost
        - *Multi-Class Classification*: Gradient Boosting with softmax - `xgboost`, `lightgbm` libs

**TODO**: Check how each of these models, bagging, boosting would be modified for Regression.

### Deep Learning - Neural Networks (NN)
- *Basics*: Weights, Biases, Support Layers, Fully-Connected Neurons, Feed-Forward NN, Fully-Connected NN
- *Activation Functions*: Step (Stair-Step, Heavyside, Unit Step, Sign), ReLU (Leaky, Shifted, Maxout, Noisy), Smoothed ReLU (Softplus, ELU, Swish), Sigmoid/Logistic, tanh
- *Activation Function Uses*: in hidden layers, output (regression, binary classify, multi-class)
- *Backpropogation*: delta, error gradient (cross entropy), Error Curve, Gradient Descent
- *Optimization*: No Free Lunch theorem
    - *Learning Rate*: decay (exponential, delayed exponential, fixed-step, error-based, bold-driver)
    - *Gradient Descent*: Batch, SGD, Mini-Batch SGD, Momentum, Nestrov momentum, Adaptive (Adagrad, Adadelta, Adam)
- *Regularization*: Dropout, Batch Normalization

### Neural Network Architectures
- *Basics*: Transfer Learning vs Fine Tuning vs Downstream models
- *CNN*: Convolution (2D, 1D, 1x1), Weight Sharing, Spatial Filter (Feature Detector), Footprint, Feature Map, Padding, Pooling (avg, max, Striding), Upsampling, best practices, real-life (MNIST Digit Classifier, VGG16), Visualizing CNN filters, Adverserial inputs
- *Autoencoder*: Semi-Supervised ML, Encoder-Decoder, Generator, Blending (content, parametric), Latent Space, Convolutional Autoencoder, Denoising, Variational Autoencoder (Reparametrization Trick, KL Divergence)
- *RNN*: 
