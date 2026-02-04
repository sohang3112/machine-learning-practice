# Chapter 1 Exercises

1. How would you define Machine Learning?
A: Systems that learn from data, instead of instructions being explicitly hardcoded.

2. Can you name four types of problems where it shines?
A: Image Classification, Text Sentiment Analysis, Language Translation, OCR (image -> text)

3. What is a labeled training set?
A: Training data has classes (req ans) of data.

4. What are the 2 most common supervised tasks?
A: Classification, Regression

5. Can you name 4 common unsupervised tasks?
A: Clustering, Anamoly Detection, PCA (Principal Component Analysis), Assocation Rules (relationships between data - eg. frequently bought together products)

6. What type of ML algo would you use to allow a robot to walk in various unknown terrains?
A: Object Detection (computer vision)

7. What type of algo would you use to segment customers into multiple groups?
A: Clustering

8. Is spam detection supervised or unsupervised learning problem?
A: Supervised

9. What is an online learning system?
A: Online/Incremental: (fast) training on a single sample or mini-batch at a time, so can adapt easily to new data.
    * real-time data (but risk of bad data contaminating model)
    * **Out-of-Core learning**: not enough RAM, so train on a batch,
    then free memory and train on next batch.

10. What is out-of-core learning?
A: see 2nd bullet point of previous answer

11. What type of learning algo relies on a similarity measure to make predictions?
A: k-Nearest Neighbours (kNN)

12. Difference between model parameter and learning algo hyperparameter?
A: Difference:
    * Model parameter / weight: learnt during training, makes inference
    * Hyperparameter: (eg. learning rate) affects how training goes

13. What do model-based learning algos search for? What is the most
    common strategy they use to succeed? How do they make predictions?
A: search for weights for which validation loss is minimized;
   gradient descent;
   (neural network) forward propogation: each layer sends output to next layer.

14. 4 main ML challenges?
A: Challenges:
    * poor data quality --> clean data beforehand
    * underfit (less data)
    * overfit (regularization or choose less complex model)
    * slow train & inference speed

15. If model performs great on train data but generalize poorly to new data, what is happening? 3 solutions?
A: Overfitting; solutions:
    * use simpler model (less params)
    * early stopping in train if val loss stops improving
    * regularization:
        * dropout (train-only) layer randomly disables some params in each train step
        * constrain weight possible values (keep small) by adding penalty to loss function

16. What is test set, why use it?
A: Data kept seperate for final evaluation before training several epochs (train & val data); so that learning, hyperparam etc. not affected by it.