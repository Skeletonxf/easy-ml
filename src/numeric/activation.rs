/*!
 * Generic activation function definitions for any [Numeric]/[Real] type.
 */

use crate::numeric::Numeric;
use crate::numeric::extra::Real;

/**
 * The [relu](https://en.wikipedia.org/wiki/Rectified_linear_unit) function
 * `max(0, x)`, for any x of a [Numeric] type.
 */
pub fn relu<T: Numeric>(x: T) -> T {
    if x > T::zero() { x } else { T::zero() }
}

/**
 * The [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) standard logistic function
 * `1 / (1 + e^-x)`, for any x of a [Real] type.
 */
pub fn sigmoid<T: Real>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}
