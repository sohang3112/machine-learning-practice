# Machine Learning & Deep Learning
## Notes & Practice Exercises
Practice Notes on Machine Learning (with code &amp; data for practice exercises). Prefer the most popular libraries - `sklearn` for ML, `pytorch` for Deep Learning.

Sometimes a combination of Deep Learning and ML techniques can be useful. For example, first a Convolutional Neural Network pre-trained on generic images (eg. ImageNet database) is used to identify image features. Then these high-level features are input to ML algo like Support Vector Machine for specific task - eg. whether a tumour is malignant or not. 

### Topics Covered - Revise

#### Machine Learning (ML)
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
- *ML Model Types*: Supervised, Unsupervised, Semi-Supervised, Reinforcement Learning (RL), Parametric vs Non-Parametric 
- *ML Basic Models*: SVM, Naive Bayes, kNN, Decision Trees, Multi-Class (One vs One, One vs Rest)
- *ML Ensemble*: Law of Diminishing Returns, Bagging, Random Forests vs Extremely Random Trees
    - *Boosting*: Weighted Plurality, Weak Learner, Decision Stump
        - *Binary Classification*: Adaboost
        - *Multi-Class Classification*: Gradient Boosting with softmax - `xgboost`, `lightgbm` libs

**TODO**: Check how each of these models, bagging, boosting would be modified for Regression.

#### Deep Learning (DL) - Neural Networks
TODO


### Books
- **TODO:** Apply theory with book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"
- [Flexible Imputation of Missing Data](https://stefvanbuuren.name/fimd)

### TODO
- [-] Try Classification with [Iris Dataset](https://www.kaggle.com/datasets/uciml/iris): available in `sklearn.datasets.load_iris()`. **WIP:** Did Principal Component Analysis to reduce feature dimensions.
- [ ] Explore ML methods with [tabular datasets](https://dagshub.com/datasets/tabular/).
- [ ] Try making adverserial inputs for famous CNNs like VGG.

### Kaggle Problems to Solve
- [ ] [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk): Probabilistic Classification
- [ ] [Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction): Probabilistic Classification
- [-] [VSB Power Line Fault Detection](https://www.kaggle.com/c/vsb-power-line-fault-detection): Binary Classification: **WIP** (trying it in Kaggle Notebook)
- [ ] [Microsoft Malware Prediction](https://www.kaggle.com/c/microsoft-malware-prediction): Binary Classification
- [ ] [IEEE Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection): Binary Classification