# Statistics Notes
Notes from reading book "Introduction to Statisitical Learning with Python".

Types of Problems:
- *Prediction*: Predict output $Y$ given input $X$. Predictive Accuracy is main concern.
- *Inference*: Find relation/effects of different parts of input $X$ on output $Y$. Interpretability is main concern.

Types of Methods:
- *Parametric*: Assume a type of function in advance - eg. line fitted with Least Squares method. More interpretable, less flexible.
- *Non-Parametric*: Type of function not assumed. Less interpretable, more flexible.

Even in pure prediction problems where we're not at all concerned with interpretability, less flexible models may still perform better due to the greater possibility of over-fitting in more flexible models.

![Flexibility vs Interpretability Tradeoff](tradeoff_flexibility_interpretability.png)

**Degrees of Freedom** quantifies the flexibility of a curve - a more restricted, smoother curve has less degrees of freedom.

As model is trained/fitted, training MSE (Mean Square Error) reduces but test MSE follows a U shape - it first decreases, but then increases due to over-fitting. **Over-fitting** is when a less flexible model would have given a smaller test MSE.

**Bias-Variance Tradeoff**: Mean Square Error (MSE) of actual output $Y$ with predicted output $f(X)$ is composed of variance and squared bias of input data, plus the variance of the irreducible error $\epsilon$. To minimize MSE, we need to choose a model that produces low variance and low bias - we can never reduce error below variance of the irreducible error.

$$MSE = \mathbb{E}[Y - f(X)] = Variance(f(X)) + Bias(f(X))^2 + Variance(\epsilon)$$

- Variance refers to amount by which estimated function $f$ would change if different training data were used. More flexible models have higher variance & more prone to over-fitting. 
- Bias refers to the error introduced due to modelling a complex real-world phenomenon with a simple function.
- *Irreducible Error* is the error due to missing inputs - i.e., inputs which affect output $Y$ but weren't included in input $X$.

Usually we don't know actual output $Y$ of test data, so we instead use *Cross Validation* to estimate Test MSE using training data.

**Bayes Classifier** predicts the class whose *conditional probability* is highest. *Bayes Decision Boundary* (curved boundary) divides the data into regions for each class. It has the lowest error rate: *Bayes Error Rate*, analogous to *Irreducible Error*. **For real data we don't know the probability distribution. So Bayes Classifier is the unattainable Gold Standard to compare all methods against.**

*k-Nearest Neigbours* identifies the K points closest to input $X$ and calculates conditional probability of each class as no. of points (in K closest points) having that class. It can be very close to optimal Bayes Classifier.