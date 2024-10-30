Just realized relation between Neural Network & Linear Regression with OLS (Ordinary Least-Squares):
- Take simplest fully-connected neural network - only input & output layers, no hidden layers, no activation
- Then each output neuron a linear combination of all input values, using weights/parameters. But that's just like Linear Regression!!
- **Difference from Linear Regression**: OLS uses a closed-form solution (a math equation gives exact, best solution for all training data).
  But SGD (Stochastic Gradient Descent) is iterative - we try to find a good solution, rarely do we find absolute best solution.
- Of course other important difference is activation function and hidden layers, which allow neural network to model non-linearities.

Using same argument, a neuron of hidden layer is also similar to a *weighted ensemble*, with main difference again being learning technique.