# Changelog

## Version 1.3.1

Fixed broken link to XOR example in documentation.

## Version 1.3

Added Forward and Reverse mode Automatic Differentiation wrapper structs.
Added example for solving the XOR problem with a neural net using Automatic
Differentiation.
Added RowMajor versions for matrix iterators
Added matrix and scalar operations to Matrix

## Version 1.2

Added a simpler Naïve Bayes example and supporting library code for
computing f1 scores.

## Version 1.1

Deprecated `Matrix::unit` and renamed to `Matrix::from_scalar`
Deprecated `Gaussian::map` and renamed to `Gaussian::probability`

Added fully worked Naïve Bayes example and supporting library code for
Gaussians and linear algebra.

Improved the explanation in some of the runtime `panic!` errors

## Version 1

Released with examples and library code for:

- Linear Regression
- k-means Clustering
- Logistic Regression
- using a custom numeric type such as `num_bigint::BigInt`
