⍝ WIP: Implement Multi-Class Classification Neural Network

⍝ Activation Functions - take layer of neurons and return result layer
relu ← (⊢×>∘0)        ⍝ or equivalently:  relu ← 0@(<∘0)
leakyrelu ← (÷∘10)@(<∘0)
softmax ← (⊢÷+/)*           

depth ← 4      ⍝ no. of layers (including output layer)
width ← 3      ⍝ no. of neurons in a single layer
X ← 3 4 5      ⍝ input vector

⍝ BUG: This is wrong - each layer has its own 2D weight matrix that feeds forward to next layer
⍝ So W should be 3D - D x W x (W+1), where D = depth, W = width, W+1 has +1 to account for bias terms
W ← ¯1+2×?width depth⍴0    ⍝ init with random weights in [-1,1]
⍝ TODO: use actual initialize algo to improve learning

⍝ leakyrelu activation in hidden layers (since it learns faster than relu), softmax activation in output layer
output          ⍝ TODO
probabs ← softmax output
⊃⍒probabs       ⍝ output class index (having max probability)