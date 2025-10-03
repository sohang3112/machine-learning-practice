# Book: Numerical Linear Algebra (by Lloyd N. Trefethen and David Bau)

This book is turning out to be quite hard, it's taken many days for me to go through just 1st chapter even without the exercises and proofs!
To try to go through faster, I'm skipping all the proofs, doing about half of the chapter exercises.

## Conventions
* capital symbols (eg. $A$) are matrices, lowercase (eg. $c$) are scalars, lowercase bold (eg. $\mathbf{x}$) are vectors.
* dealing with complex numbers / spaces / vectors / matrices here -- $\mathbb{C}$ is complex numbers space.
* **Basis** of vector space is a set of linearly independent vectors that "span" the space, i.e., any vector can be expressed as linear combination of basis vectors.
  They are perpendicular to each other, i.e., dot product of any 2 basis vectors is 0.
  They DON'T necessarily need to be of unit magnitude.
  Unit basis vectors are denoted as $\mathbf{e_1}$, $\mathbf{e_2}$, $\mathbf{e_3}$, etc. - 
  these correspond to $\mathbf{i}$, $\mathbf{j}$ and $\mathbf{k}$ in physics 3D space.

## Revision (prior concepts)

**Some formulae revised:**
- matrix product transpose: $(A B)^T = B^T A^T$ 

**Some extra notes NOT from book:**
* matrix multiplication is associative ( $A (BC ) = (A B) C$ ) but not commutative ( $A B \neq B A $).
* "dot/inner product" of complex numbers $z$ and $w$ is $Re(z \overline{w})$, 
   i.e., real part of Hermitian inner product $z \overline(w)$ : https://math.stackexchange.com/a/2699556/992184
* Diagonal matrix $diag(a,b..)$ denotes matrix having given elements on diagonal, 0 elsewhere. 
  Transpose of a diagonal matrix leaves it unchanged.
  Eg. Identity Matrix $I = diag(1, 1 \cdots 1)$ 
* [**Gaussian elimination method**](https://www.geeksforgeeks.org/dsa/gaussian-elimination/): Solve $A \mathbf{x} = \mathbf{b}$ by converting to *row-echelon matrix* (ie upper triangular), then start from last and back-substitute variables.
  Eg. of row-reduced matrix:
  $$
  \begin{matrix} 
  a & b & c \\
  0 & d & e \\
  0 & 0 & f
  \end{matrix}
  $$

* **Image of matrix** $A$ is space of all possible vectors you can get on applying $A \mathbf{x}$: https://www.projectrhea.org/rhea/index.php/Image_(linear_algebra)
- Kernel of matrix -- TODO (didnt understand); image & kernel related in **Rank Nullity Theorem** .

## Parts

1. Fundamentals
    1. [Matrix-Vector Multiplication](Lecture1.md)
    2. [Orthogonal Vectors and Matrices](Lecture2.md)
