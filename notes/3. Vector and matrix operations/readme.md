# Vector and matrix operations

## Vector

- Array of numbers

![Untitled](0.png)

### Scalar operations

- Involve a vector and a number
- Addition/substraction/multiplication/division

![Untitled](1.png)

#### Vector addition/substraction

- Two vectors must have the same dimension
- Element-wise operation

![Untitled](2.png)

How it works

![Untitled](3.png)

Example

#### Dot product

- Two vectors involved
- Results in a scalar

![Untitled](4.png)

How it's done

![Untitled](5.png)

Example

### Revisiting the notation of the artificial neuron

![Untitled](6.png)

![Untitled](7.png)

*h* can be rewritten as the dot product

- Using linear algebra for NN notation is much more elegant and clearer, that's why it's commonly used.

## Matrices

- Rectangular grid of numbers (like a spreadsheet)

![Untitled](8.png)

*i* represents the row indices, *j* represents the column indices

### Matrix dimensions

- Dimensions indicated by # or rows and columns

![Untitled](9.png)

### Row and column vectors

- Row vector = (1,n) matrix
- Column vector = (n,1) matrix

![Untitled](10.png)

Vectors can be thought of as matrices

### Matrix transposition

- Switching rows and columns

![Untitled](11.png)

![Untitled](12.png)

A simple/condensed manner of notating this

### Scalar operations

- Addition/substraction/multiplication/division of vector with a number

![Untitled](13.png)

### Matrix addition/substraction

- Matrices must have same direction
- Element-wise operation

![Untitled](14.png)

### Matrix multiplication

- \# of columns of the 1st matrix must be equal to # of rows of the 2nd
- Product of an (m,n) matrix and a (n,k) matrix is an (m,k) matrix

![Untitled](15.png)

The resulting matrix has the rows of the 1st and the columns of the 2nd 

![Untitled](16.png)

How the multiplication is performed

![Untitled](17.png)

Another way to understand matrix multiplication

## Conclusion

We can use all these notations to see how a neural network performs it's computations, which will be the next topic.