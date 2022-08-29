/*!
Linear regression examples

[Overview](https://en.wikipedia.org/wiki/Linear_regression).

# Linear regression to fit a polynomial line
The code below is a method for [minimising the sum of squares error](https://en.wikipedia.org/wiki/Least_squares)
in choosing weights to learn to predict a polynomial ine. The method creates a design matrix
from the inputs x, expanding each row from \[x\] to [1, x, x^2] to allow the model
to represent the non linear relationship between x and y. To model more complex
x and f(x), more complex basis functions are needed (ie to model a n-degree polynomial
you will probably need n + 1 polynomial basis functions from x^0 to x^N).

This example does not include any methods to prevent overfitting. In practise you
may want to use some kind of [regularisation](https://en.wikipedia.org/wiki/Regularization_(mathematics))
and or holding back some data for verification to stop updating the the model when it starts
performing worse on unseen data.

## Matrix APIs

```
use easy_ml::matrices::Matrix;

// first create some data to fit a curve to
let x: Matrix<f32> = Matrix::column(
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
// we are going to fit a polynomial curve of x^2, but add the sin of x to each value for
// y to create some deteriministic 'noise'.
let y = x.map(|x| x.powi(2) + x.sin());
println!("{:?}", &y);

// Now we create a quadratic basis where each row is [1, x, x^2]
let mut X = x.clone();
// insert the 1 values as the first column, so that each row becomes [1, x]
X.insert_column(0, 1.0);
// insert another column with x^2 as the last, so that each row becomes [1, x, x^2]
X.insert_column_with(2, x.column_iter(0).map(|x| x * x));
println!("{:?}", &X);

// now we compute the weights that give the lowest error for y - (X * w)
// by w = inv(X^T * X) * (X^T * y)
// Note the use of referencing X, w, and y so we don't move them into
// a computation.
// Because we're doing linear regression and creating the matrix we take the inverse
// of in a particular way we don't check if the inverse exists here, but in general
// for arbitary matrices you cannot assume that an inverse exists.
let w = (X.transpose() * &X).inverse().unwrap() * (X.transpose() * &y);
// now predict y using the learned weights
let predictions = &X * &w;
// compute the error for each y and predicted y
let errors = &y - &predictions;
// multiply each error by itself to get the squared error
// and sum into a unit matrix by taking the inner prouct
// then divide by the number of rows to get mean squared error
let mean_squared_error = (errors.transpose() * &errors).get(0, 0) / x.rows() as f32;

println!("MSE: {}", mean_squared_error);
assert!(mean_squared_error > 0.41);
assert!(mean_squared_error < 0.42);
println!("Predicted y values:\n{:?}", &predictions);
println!("Actual y values:\n{:?}", &y);

// now we have a model we can predict outside the range we trained the weights on
let test_x: Matrix<f32> = Matrix::column(vec![-3.0, -1.0, 0.5, 2.5, 13.0, 14.0]);
let test_y = test_x.map(|x| x.powi(2) + x.sin());
let mut test_X = test_x.clone();
test_X.insert_column(0, 1.0);
test_X.insert_column_with(2, test_x.column_iter(0).map(|x| x * x));

// unsurprisingly the model has generalised quite well but
// did better on the training data
println!("Unseen x values:\n{:?}", test_x);
println!("Unseen y predictions:\n{:?}", &test_X * &w);
println!("Unseen y actual values:\n{:?}", test_y);
let errors = &test_y - (&test_X * &w);
let mean_squared_error = (errors.transpose() * &errors).get(0, 0) / test_x.rows() as f32;
println!("MSE on unseen values: {}", mean_squared_error);
assert!(mean_squared_error < 1.0);
assert!(mean_squared_error > 0.99);
```

## Tensor APIs

```
use easy_ml::tensors::Tensor;

// first create some data to fit a curve to
let x: Tensor<f32, 1> = Tensor::from(
    [("row", 13)],
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
);
// we are going to fit a polynomial curve of x^2, but add the sin of x to each value for
// y to create some deteriministic 'noise'.
let y = x.map(|x| x.powi(2) + x.sin());
println!("{:?}", &y);

// Now we create a quadratic basis where each row is [1, x, x^2]
//let mut X = unimplemented!(); // TODO: Need an API to increase the dimensionality along a dimension for Tensors/views
```

*/
