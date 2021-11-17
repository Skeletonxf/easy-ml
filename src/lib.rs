/*!
 * If this is your first time using Easy ML you should check out some of the examples
 * to get an overview of how to use matrices then study the
 * [Matrix](matrices::Matrix) type for what you need.
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
 * - [Linear Algebra](linear_algebra)
 * - [Distributions](distributions)
 * - [(Automatic) Differentiation](differentiation)
 * - [Numerical type definitions](numeric)
 *
 * # Miscellaneous
 * - [Web Assembly](web_assembly)
 * - [Using custom numeric types](using_custom_types)
 */

pub mod differentiation;
pub mod distributions;
pub mod linear_algebra;
pub mod matrices;
pub mod numeric;
#[allow(dead_code)]
mod tensors;

// examples
pub mod k_means;
pub mod linear_regression;
pub mod logistic_regression;
pub mod naive_bayes;
pub mod neural_networks;
pub mod using_custom_types;
pub mod web_assembly;
