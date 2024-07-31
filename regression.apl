⍝ Convention: functions start with first letter in lower-case, Variables in upper-case
⍝ Run in Dyalog APL

avg ← +/÷≢         ⍝ Monadic: Average
var ← 2+.*⍨⊢-avg   ⍝ Monadic: Variance
std ← 0.5*⍨var     ⍝ Monadic: Standard Deviation
mse ← {(≢⍵)÷⍨(⍺-⍵)+.*2}   ⍝ Dyadic: Mean Square Error of predicted vs actual
z ← {⍺←⍵ ⋄ (⍺-avg ⍵)÷std ⍵}   ⍝ Dyadic: Z Score of ⍺ wrt ⍵ (vector); Monadic: calcs Z Scores of ⍵ itself
lincorr ← {(≢⍵)÷⍨⍺((+.×)⍥z)⍵}    ⍝ Dyadic: Linear Correlation Coefficient

⍝ BUG: wrong answer
⍝ return regression line (slope, intercept) after fitting on train data (X, Y)
leastsquares ← {
    (ax sx ay sy) ← ⍺ ,⍥(avg,std) ⍵       ⍝ ax,ay <- Avg of X,Y; sx,sy <- Std.Dev. of X,Y
    r ← (≢⍵)÷⍨((⍺-ay)÷sy)+.×(⍵-ax)÷sx     ⍝ linear correlation coefficient
    m ← r×sy÷sx     ⍝ slope
    m,ay-m×ax       ⍝ return slope, intercept
}

X ← 5?5
Y ← 100+X+?0⍴5
m,c ← X leastsquares Y     ⍝ regression line (slope, intercept)
Y mse (m×X+c)              ⍝ check Mean Square Error on training data only