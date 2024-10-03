# Dealing with Imbalanced Data
Source: [PyData talk](https://youtu.be/6M2d2n-QXCc?si=8jlzCyy0mB65_STF)

Situation: we're training a model to classify tumours as benignant or malicious. But we have 9:1 data imbalance!

1. Data Preprocessing options:
    - Under-Sampling: randomly sample 1K of class B to equalize data counts. *But this discards data!*
    - Generate synthetic data for under-represented class using a generator model to make up difference.
    - Ensemble: train 10 models, each using 1K of class B. Inference with majority voting.
    - Over-Sampling: Sample With Replacement class A to grow 1K -> 10K data. *But it's computationally expensive, complex algorithms*.

2. Adjusting Weights themselves for data imbalance
    - XGBoost classifier - set `scale_pos_weight` to 10K / 1K = 10. XGBoost will internally do oversampling techniques to account for the imbalance in data.
    - `sklearn` Random Forest - set `class_weight` parameter to a dict (keys -> classes, values -> weights).

3. Library (part of sklearn contrib): [imblearn](https://github.com/scikit-learn-contrib/imbalanced-learn)
    - it has classes like `BalancedBaggingClassifier`, `BalancedRandomForestClassifier` to handle imbalanced datasets.
    - also has functions to undersample, oversample
    - it can also *generate balanced batches for tensorflow and keras*.
    - also has functions to evaluate with imbalanced data.

4. Evaluation Metrics: *don't* use accuracy since it's misleading. Due to imbalanced data, a model could learn to always give the same result and still get 99% accuracy!
    - Preferred metric is F1 Score (higher is better!) - it's Harmonic Mean of precision, recall, so F1 score is high only when both precision and recall are high
    - *Setting `scale_pos_weight` in sklearn random forest can have better F1 score!!*

5. Custom Loss Function: normally we use *Cross Entropy* loss fn in deep learning, but that may not work well for imbalanced data.
    - Focal Loss: down-weights the well-classified samples, puts more emphasis on under-represented class.
        so it will trade away some performance in over-represented class for better performance on minority class.

6. Things to keep in mind:
    - Maintain proportions while train-test split:
        In sklearn train_test_split fn, add `stratify=y` keyword arg so that in the split data, each dataset maintains the same proportion of data in each class as original.
    - If some classes have very few data (say, less than 10 in total 1000 data) - just remove them, they will only confuse the model.