# Part 1, Lecture 2. Orthogonal Matrices and Vectors

* *Complex Conjugate*: $Im(z^*) = -Im(z)$
* **Adjoint** / *Hermitian Conjugate*: If $B^*$ is adjoint of $A$, then $b_{ji} = a_{ij}^*$, i.e. complex conjugates with rows and columns exchanged. *Equivalent to Transpose for real matrix*. Converts column to row vector.
    * Basically do conjugate of all matrix values, then transpose. That's also how you do it in numpy: `A.conj().T`.
    * If square matrix $A = A^*$, then it is a **Hermitian Matrix**.
* **Inner Product of Vectors Having Complex Numbers**: $\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^* \mathbf{y} = \sum x_{ij}^* y_{ij}$ -- note it's NOT symmetric for complex vector, took conjugate / row vector of first arg but not second.
    * **Outer Product** is opposite order (second arg is conjugate / row vector); $\mathbf{x}$ has size $m$, $\mathbf{y}$ has size $n$: 
        $$
        \mathbf{x} \otimes \mathbf{y} = 
        \mathbf{x} \mathbf{y}^* =
        \begin{bmatrix}
        x_1 y_1 & x_1 y_2 & \cdots & x_1 y_n \\
        x_2 y_1 & x_2 y_2 & \cdots & x_2 y_n \\
        \vdots & \vdots & \ddots & \vdots \\
        x_m y_1 & x_m y_2 & \cdots & x_m y_n
        \end{bmatrix}
        $$
    * Basically inner product leads to a scalar: $(1,n) \cdot (n,1) --> (1,1)$ whereas outer product to a matrix: $(m,1) \cdot (1,n) --> (m,n)$
* *Euclidean length of vector* is square root of inner product of vector with itself: $|\mathbf{x}| = \sqrt{\mathbf{x} \cdot \mathbf{x}}$.
* Angle between 2 vectors $cos(\alpha) = \frac{\mathbf{x} \cdot \mathbf{y}}{|\mathbf{x}| |\mathbf{y}|}$
* $(A B)^* = B^* A^*$ and $(A B)^{-1} = B^{-1} A^{-1}$
    * Inverse of conjugate = conjugate of inverse, so used with shorthand notation $A^{-*} = (A^{-1})^* = (A^*)^{-1}$
* *Orthogonal / Perpendicular vectors* have dot product zero: $\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^* \mathbf{y} = 0$ 
* *Orthogonal Set* has mutually orthogonal vectors - they're linearly independent. If all are unit vectors, then it's called a **Orthonormal Set**.
* **Decomposing Vector Components** using a set of orthonormal vectors $\{\mathbf{q_1}, \mathbf{q_2} \cdots \mathbf{q_n}\}$ (note that if it covers whole vector space, i.e., become *unit basis vectors*, then residual scalar $r = 0$):
    * In form of inner product (scalar) $\mathbf{q_i}^* \mathbf{v}$:
    $$\mathbf{v} = r + \sum_{i=1}^{n} (\mathbf{q_i}^* \mathbf{v}) \mathbf{q_i}$$
    * In form of outer product matrix $\mathbf{q_i} \mathbf{q_i}^*$:
    $$\mathbf{v} = r + \sum_{i=1}^{n} (\mathbf{q_i} \mathbf{q_i}^*) \mathbf{v}$$
* **Unitary Matrix:** for unitary matrix $Q$, $Q^* = Q^{-1}$, i.e., $Q Q^* = I$. If unitary matrix $Q$ is also real, then it's **Orthogonal**.
    * Inner Products of vectors are preserved: $(Q \mathbf{x})^* (Q \mathbf{y}) = x^* y$
    * from above, vector magnitudes and angle between them also preserved: $|Q \mathbf{x}| = |\mathbf{x}|$
* *Pythagoras Theorem (extended)*: for $n$ orthogonal vectors $\mathbf{x_1} \cdots \mathbf{x_n}$ :
  $$ |\sum \mathbf{x_i}|^2 = \sum |\mathbf{x_i}|^2 $$

Some properties:
* If a matrix is both triangular and unitary, then it's a diagonal matrix.
* All eigenvalues of a Hermitian matrix are real.
* Eigen-vectors of distinct eigen values are orthogonal.

## Exercises for Lecture 2

TODO: unverified, actually check these answers!!

- *2.1* $A$ has either upper or lower triangle all 0s. 
        It's unitary $A = A^*$, so $upper_triangle^* = lower_triangle$ - but either LHS or RHS is 0, so both sides must be 0.
        So $A$ is diagonal matrix (all 0s except for main diagonal).
- *2.2*, *2.3*: proofs, skipped
- *2.4*: $A \mathbf{x} = \lambda \mathbf{x}$ - unitary, so $A^{-1} = A^*$, or $A = A^{*-}$ ; 
         $A^{*-}$ will eigen value $1 / \lambda$ due to inverse ; 
         but since $A$ is equal to it, so $\lambda = 1 / \lambda$ - only eigen value satisfying this is 1.
- *2.5*, *2.6*, *2.7*: proofs, skipped