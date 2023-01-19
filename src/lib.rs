/*!
 * If this is your first time using Easy ML you should check out some of the examples
 * to get an overview of how to use matrices or tensors then check out the
 * [Matrix](matrices::Matrix) type or [Tensor](tensors::Tensor) type for what you need.
 *
 * `Matrix` is a straightforward 2 dimensional matrix with APIs built around the notion of
 * rows and columns; `Tensor` is a named tensor with full API support for 0 to 6 dimensions.
 * Naturally, a 2 dimensional tensor is also a matrix, but the APIs are more general so may
 * be less familiar or ergonomic if all you need is 2 dimensional data.
 *
 * # Examples
 * - [Linear Regression](linear_regression)
 * - [k-means Clustering](k_means)
 * - [Logistic Regression](logistic_regression)
 * - [Naïve Bayes](naive_bayes)
 * - [Neural Network XOR Problem](neural_networks)
 *
 * # API Modules
 * - [Matrices](matrices)
 * - [Named tensors](tensors)
 * - [Linear Algebra](linear_algebra)
 * - [Distributions](distributions)
 * - [(Automatic) Differentiation](differentiation)
 * - [Numerical type definitions](numeric)
 *
 * # Miscellaneous
 * - [Web Assembly](web_assembly)
 * - [SARSA and Q-learning using a Matrix for a grid world](sarsa)
 * - [Using custom numeric types](using_custom_types)
 */

pub mod differentiation;
pub mod distributions;
pub mod interop;
pub mod linear_algebra;
pub mod matrices;
pub mod numeric;
pub mod tensors;

// examples
pub mod k_means;
pub mod linear_regression;
pub mod logistic_regression;
pub mod naive_bayes;
pub mod neural_networks;
pub mod sarsa;
pub mod using_custom_types;
pub mod web_assembly;
