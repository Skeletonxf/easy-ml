/*!
 *
 * TODO
 */

use crate::numeric::{Numeric, NumericRef};
use crate::numeric::extra::{Real, RealRef};
use std::ops::{Add, Sub, Mul, Neg, Div};

/**
 * A dual number which traces a real number and keeps track of its derivative.
 * This is used to perform Automatic Differentiation, as in no getting out the paper
 * to do calculus with algebra!
 *
 * Trace implements only first order differentiation. For example, given a function
 * 3x<sup>2</sup>, you can use calculus to work out that its derivative with respect
 * to x is 6x. You can also take the derivative of 6x with respect to x and work out
 * that the second derivative is 6. By instead writing the function 3x<sup>2</sup> in
 * code using Trace types as your numbers you can compute the first order derivative
 * for a given value of x by passing your function `Trace { number: x, derivative: 1.0 }`.
 *
 * ```
 * use easy_ml::differentiation::Trace;
 * let x = Trace { number: 3.2, derivative: 1.0 };
 * let dx = Trace::constant(3.0) * x * x;
 * assert_eq!(dx.derivative, 3.2 * 6.0);
 * ```
 */
#[derive(Debug)]
pub struct Trace<T> {
    /**
     * The real number
     */
    pub number: T,
    /**
     * The first order derivative of this number.
     */
    pub derivative: T
}

impl <T: Numeric> Trace<T> {
    /**
     * Constants are lifted to Traces with a derivative of 0
     */
    pub fn constant(c: T) -> Trace<T> {
        Trace {
            number: c,
            derivative: T::zero(),
        }
    }

    /**
     * To lift a variable that you want to find the derivative of
     * a function to, the Trace starts with a derivative of 1
     */
    pub fn variable(x: T) -> Trace<T> {
        Trace {
            number: x,
            derivative: T::one(),
        }
    }

    /**
     * Computes the derivative of a function with respect to x.
     *
     * This is a shorthand for `(function(Trace::variable(x))).derivative`
     */
    pub fn derivative(function: impl Fn(Trace<T>) -> Trace<T>, x: T) -> T {
        (function(Trace::variable(x))).derivative
    }
}

/**
 * Any trace of a Cloneable type implements clone
 */
impl <T: Clone> Clone for Trace<T> {
    fn clone(&self) -> Self {
        Trace {
            number: self.number.clone(),
            derivative: self.derivative.clone(),
        }
    }
}

/**
 * Any trace of a Copy type implements Copy
 */
impl <T: Copy> Copy for Trace<T> { }

/**
 * Any trace of a PartialEq type implements PartialEq
 */
impl <T: PartialEq> PartialEq for Trace<T> {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number && self.derivative == other.derivative
    }
}

/**
 * Elementwise addition for two referenced traces of the same type.
 */
impl <T: Numeric> Add for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn add(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone() + rhs.number.clone(),
            derivative: self.derivative.clone() + rhs.derivative.clone(),
        }
    }
}

/**
 * Elementwise addition for two traces of the same type.
 */
impl <T: Numeric> Add for Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn add(self, rhs: Trace<T>) -> Self::Output {
        &self + &rhs
    }
}

/**
 * Elementwise addition for two traces with one referenced
 */
impl <T: Numeric> Add<&Trace<T>> for Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn add(self, rhs: &Trace<T>) -> Self::Output {
        &self + rhs
    }
}

/**
* Elementwise addition for two traces with one referenced
*/
impl <T: Numeric> Add<Trace<T>> for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn add(self, rhs: Trace<T>) -> Self::Output {
        self + &rhs
    }
}

/**
 * Elementwise multiplication for two referenced traces of the same type.
 */
impl <T: Numeric> Mul for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn mul(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone() * rhs.number.clone(),
            derivative: (self.derivative.clone() * rhs.number.clone())
                + (self.number.clone() * rhs.derivative.clone()),
        }
    }
}

/**
 * Elementwise multiplication for two traces of the same type.
 */
impl <T: Numeric> Mul for Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn mul(self, rhs: Trace<T>) -> Self::Output {
        &self * &rhs
    }
}
