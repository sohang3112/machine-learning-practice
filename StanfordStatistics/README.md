# Statistics Notes
Notes from course [Intro to Statistics by Stanford](https://www.coursera.org/learn/stanford-statistics). All material from the course is in OneDrive folder "Stanford-Statistics-Course-Slides".

**TODO:** Failed quiz of [Module 3](#probability) mainly due to Bayes' Formula wrong answer - try it again.

**Probability Density of Continous Data**: In a continous probability historgram, absolute probability at any single point is 0, since there are infinite number of points. So we instead use Probability Density, which is probability per unit length.

### Variance Formula
$$Variance = \sigma^2 = \frac{\sum (x - \bar{x})^2}{n}$$
where $\bar{x}$ is the Mean, $\sigma$ is Standard Deviation, $N$ is no. of observations.

It can also be restated in terms of Expected Value:
$$Variance = \sigma^2 = E[x^2] - E[x]^2$$

$$WeightedVariance = \sigma^2 = \frac{\sum (x - \bar{x})^2 \cdot w_x}{\sum w_x}$$
where $w_x$ is the weight corresponding to each x.

## Descriptive Statistics
Purpose:
- Communicate info
- Support reasoning

### Data Visualization
Discrete Data:
- Pie Chart & Bar Graph (discrete data frequencies), Scatter Plot (for 2D data)
- Dot Plot: bar graph put on its side (Y,X <- classes, frequencies) with dots instead of bars:

![Dot Plot](images/stats_dot_plot.png)

- Variable Width Histogram: $BarArea = Frequency$  (relative frequency can also be used)
    - Bar width = size of range (all bars have different sized ranges)
    - $BarHeight = Density = Frequency / BarWidth$

![Variable Width Histogram](images/variable_width_histogram.png)

- Boxplot / Box-and-Whisker Plot: shows (in order): min, 25% percentile (1st quartile), median, 75% percentile (3rd quartile), max

![Boxplot](images/boxplot.png)

Boxplots are compact so allow easily comparing multiple datasets - eg. boxplots of weather for multiple days:

![Boxplot Comparision](images/boxplots_comparision.png)

**Data Visualization Pitfalls**: Many sophisticated plots can look visually appealing but be misleading & make data hard to interpret - eg. a 3D perspective bar graph where 3D volume of bars is shown but in fact volume isn't relevant at all (only bar's height is), and also 0 of each bar is at a different level making the plot hard to understand.

### Numerical Measures
- In a symmetric histogram, Mean = Median. But when histogram is very skewed, Median is preferred (Mean is much smaller or larger than Median).
- **Interquartile Range** measures how spread the data is: *3rd quartile (25% percentile) - 1st quartile (75% percentile)*
- Mean, Std. Deviation are both sensitive to a few small / large outliers. If that's a concern, use Interquartile Range instead.


## Producing Data and Sampling
While making a sample (to make inferences about population based on sample), avoid biases:
- *Selection Bias*: Sample of Convinience - eg. sample of people from hometown is a poor representative of whole country
- *Non-Response Bias*: parents less likely to respond to surveys at 6pm because busy with children
- *Voluntary-Response Bias*: people with very good or very bad experiences are much more likely to leave reviews.

Sampling Designs:
- *Simple Random Sample*: Random without Replacement
- *Stratified Random Sample*: Divide population into strata (eg. based on income), then draw from each using simple random.

$$Estimate = Actual + Bias + Chance/Sample Error$$
where:
- Chance/Sample Error gets smaller as sample size increases. Changing sample changes this error value.
- But bias isn't eliminated by increasing sample size.

An *Observational Study* may find a correlation between 2 things, but to estabilish causation a **Randomized Control Experiment** needs to be done:
- *Control* group gets placebo, *Treatment* group gets actual treatment - the groups should be otherwise identical.
- *Double-Blind experiment*: Neither subjects nor evaluators should know who is in which group.


## Probability
If A and B are mutually exclusive events, then:
- P(A or B) = P(A) + P(B)
- P(A and B) = P(A) P(B)

Conditional Probability: $P(B|A) = \frac{P(A and B)}{P(A)}$

Bayes' Rule: $P(B|A) = \frac{P(A|B) P(B)}{P(A)} = \frac{P(A|B) P(B)}{P(A|B) P(B) + P(A|~B) P(~B)}$


## Normal Approximation for Data & Binomial Distribution
*Three Sigma Rule*: In Normal Distribution:
- 68% (about 2/3) data falls within one standard deviation from mean
- 95% in 2 standard deviations
- 99.7% in 3 standard deviations

**Z-Score / Standardized Value**: Using $\bar{x}$ as Mean, $s$ as Standard Deviation:
$$z1 = \frac{x1 - \bar{x}}{s}$$ 

Z Score has no units. $z = 2$ means x is 2 standard deviations above average. $z = -3$ means x is 3 standard deviations below average. *Y axis (frequency count) has no role in calculating Z Score*.

This standard value forms a **Standard Normal Curve** with mean 0, variance 1:

![Standard Normal Curve](images/standard_normal_curve.png)

### Normal Approximation
Estimating area under an approx normal curve.

A [*Z Table / Normal Table*](https://z-table.com/) tells the *percentile area* to the left of a Z Score.

To find area between points $x1$ and $x2$ on a normal curve:
- Calc Z Scores: $z1 = \frac{x1 - \bar{x}}{s}$, $z2 = \frac{x2 - \bar{x2}}{s}$
- Lookup percentile values for these Z Scores using Z Table: $p1$ % and $p2$ %.
- Find *percentile area*: $(p2-p1)$ %.

*X (given percentile)*: $x1 = \bar{x} + z1 \times s$

### Binomial Probability
Binomial Probability is ${}^n \mathrm{ C }_k p^k (1-p)^{n-k}$ where $p$ is the probability of success and $n$ is the no. of independant experiments. 

Binomial Coefficient is: ${}^n \mathrm{ C }_k = \frac{n!}{k! (n-k)!}$

The no. of successes $X$ is a **Random Variable** and has the Binomial Distribution. Probabilities of various $X$ can be visualized with a **Probabilistic Histogram**:

![Probability Historgram](images/probability_historgram.png)

#### Approximating Binomial Probab with Normal Distribution
As $n \to \inf$, the binomial probability distribution approaches normal distribution. We can use this to approximate Binomial Probability of at most $x$ successes with [Normal Approximation](#normal-approximation) using approx mean $\bar{x} = np$, std. dev. $s = \sqrt{np(1-p)}$. These formulae are derived [using Central Limit Theorem](#deriving-normal-probability-distribution-formulae).

**Example**: If p = P(win a small prize) = 0.2, we play n = 50 times. What is P(at most 12 small prizes) ?
- Mean $\bar{x} = np = 50 \times 0.2 = 10$
- Std. Dev. $s = \sqrt{np(1-p)} = \sqrt{50 \times 0.2 \times (1-0.2)} = \sqrt{8} \approx 2.8$
- Z Score $z = \frac{x - \bar{x}}{s} = \frac{12 - 10}{2.8} \approx 0.71$
- P(at most 12 small prizes) = 0.7611 (by looking up 0.71 in [Z Table](https://z-table.com/))

**Simple Random Sample**: Sampling Without Replacement: It's not same as Binomial Probability because probability $p$ changes after each experiment. But if $n >> k$, then we can approximate it as Binomial Probability (& therefore Normal Distribution).


## Sampling Distributions and Central Limit Theorem
Estimating Population characterstics based on Sample:
- *Parameter*: what we want to estimate - population mean $\mu$, std. dev. $\sigma$
- *Statistic (estimate)*: sample characterstic based on which we want to estimate population characterstic - sample mean $\bar{x}$, std. dev. $s$

**Standard Error (SE)** tells us how far away a statistic will be from its *Expected Value*. It's simply the standard deviation of the statistic.

**Expected Value of 1 random draw**: population mean $\mu$ (give or take population std. dev. $\sigma$)

### Drawing multiple times 
We draw from population $n$ times. Standard Error becomes smaller if we use a larger sample size $n$ - it  doesn't depend on size of population, only sample size. *Standard Error of a statistic is simply Standard Deviation of the statistic.*

#### Mean of n draws
- Expected Value $E[\bar{x_n}]$ = population average $\mu$
- Standard Error $SE[\bar{x_n}] = \frac{\sigma}{\sqrt{n}}$

#### Sum of n draws
- Relation to mean: $S_n = n \bar{x_n}$
- Expected Value $E[S_n] = n \mu$
- Standard Error $SE[S_n] = \sqrt{n} \sigma$ - so variability of sum of $n$ draws increases at rate $\sqrt{n}$.

### Expected Value & Standard Error for Percentages
A question like "what % of voters voted in election" is asking for the mean.
- assign 1 to all voters who voted, 0 otherwise
- then no. of voters who voted = sum of counts of all voters $S_n$
- Question becomes: what % of labels are 1s = $\frac{S_n}{n} \times 100\%$
- Expected Value $E[\%Of1s] = \mu \times 100\%$ where $\mu$ is the population average - i.e., what % are 1s
- Standard Error $SE[\%Of1s] = \frac{\sigma}{\sqrt{n}} \times 100\%$ where $\sigma$ is std. dev. of population of 1s and 0s

Above 2 formulae (expected error & standard error) are for Sample with Replacement, but we can use them for Sample without Replacement if sample size is much smaller than population size.

#### Using Probability Histogram
The above formulae still apply when the values are **simulated** (generated from a probability histogram):
- Discrete data:
    - Mean is weighted avg (where weights are probabilities): $\mu = \sum x \cdot P(x)$ where $P(x)$ is probability of a given x.
    - Variance is weighted variance (first see [variance formula](#variance-formula)): 
    $\sigma^2 = \sum (x - \mu)^2 \cdot P(x)$
- Continous data: in terms of probability density $f(x)$:
    - Mean $\mu = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$
    - Variance $\sigma^2 = \int_{-\infty}^{\infty} (x - \mu)^2 \cdot f(x)$

#### Example
- **Question**: If a fair coin is tossed 100 times, how many tails would you expect, give or take how much?
- **Answer**: Discrete data (2 values: Head, Tail), $n=100$ repetitions of fair coin with probability $p = P(Tail) = 1/2$
    - For each coin toss, treat Tail as 1, Head as 0.
    - Mean $\mu = \sum x \cdot P(x) = 0 \cdot 1/2 + 1 \cdot 1/2 = 1/2$
    - Variance $\sigma^2 = \sum (x - \mu)^2 \cdot P(x) = (0 - 1/2)^2 \cdot 1/2 + (1 - 1/2)^2 \cdot 1/2 = 1/4$
    - Standard Deviation $\sigma = \sqrt{1/4} = 1/2$
    - Expected Value $E(S_n) = n \cdot \mu = 100 \cdot 1/2 = 50$
    - Standard Error $SE(S_n) = \sqrt{n} \cdot \sigma = \sqrt{100} \cdot 1/2 = 5$
    - **So we expect 50 tails, give or take 5.**
    - $SE(S_n)$ is simply standard deviation of $S_n$, so if this experiment were to be repeated many times, no. of tails would be between 45 and 50 (i.e., one standard deviation from mean) 68% of the time.

### Sampling Distribution
If statistic of interest is no. of tails $S_n$, then $S_n$ is a random variable whose probability distribution is given by the binomial distribution. This is called **Sampling Distribution**.

3 historgrams in Sampling Distribution:
- Probability Histogram of each coin toss:

![Probability Histogram of each coin toss](images/coin_probab_histogram.png)

- *Emperical Histogram* of no. of 100 observed tails (real data):

![Emperical Histogram](images/emperical_probab_histogram.png)

- **Population Histogram**: Probability Histogram of statistic $S_{100}$ (no. of tails):

![Probability Historgram of S_n](images/probab_histogram_sn.png)

NOTE: If we just say "Sampling Distribution histogram", then we mean Probability Histogram of the sample statistic we're looking at.

**Law of Large Numbers**: As sample size $n$ approaches population size $N$, sample mean $\bar{x}$ approaches population mean $\mu$:
$$\lim_{n \to N} \bar{x} = \mu$$

Law of Large Numbers applies to averages and percentages, **but doesn't apply to sums** as sums *increase* with size. It applies to Sampling with Replacement - **it won't work for Sample Without Replacement for large sample size (wrt population size)**.

*Advanced Law of Large Numbers*: As sample size grows larger, its emperical histogram approaches population's histogram.

### Central Limit Theorem
Sampling with Replacement: As no. of independant events $n$ gets larger, Binomial Probability approaches Normal Curve.

So when Sampling with Replacement and large sample size, probability histogram of sample statistic (avg / sum) approaches Normal Curve with mean as Expected Value, std. deviation as Standard Error.

**Sampling statistic's probability distribution is Normal, irrespective of Population Histogram**.

#### Deriving Normal Probability Distribution formulae
Recall that in [Exp. Val & Std. Error for Percentages](#expected-value--standard-error-for-percentages),
we created a distribution with values only 1s and 0s (1 if condition is met, 0 otherwise).

Let $Y_i$ be the result of a single binomial experiment - its value is either 0 or 1. Its probability of success (1) is $P(Y_i) = p$.
- Mean $E[Y_i] = 0 \cdot (1-p) + 1 \cdot p = p$
- Variance:
    - $Var[Y_i] = E[Y_i^2] - E[Y_i]^2$
    - Since $Y_i$ only takes 2 values 0,1, therefore $E[Y_i^2] = E[Y_i]^2 = p$
    - So $Var[Y_i] = p - p^2 = p(1-p)$

Let random variable $X$ be the Binomial Probability after $n$ experiments. Using Central Limit Theorem, $X$ is normally distributed for large $n$.

So from above, we get the formulae mentioned in [Approximating Binomial Probab with Normal Distribution](#approximating-binomial-probab-with-normal-distribution) using [Sum of n draws formulae](#sum-of-n-draws):
- Expected Value of Binomial Probab $E[X] = n \cdot E[Y_i] = np$
- Standard Error of Binomial Probab $SE[X] = \sqrt{n} \cdot \sigma_X = \sqrt{np(1-p)}$

#### Requirements for applying Central Limit Theorem
For normal approximation to work, requirements are:
- Either Sample with Replacement, or simulate independant variables from same distribution.
- Statistic of interest is a sum (average and percent are sums in disguise).
- Larger the skew in population histogram, larger the required sample size $n$. 
  If there's no skew, $n \ge 15$ is ok.














