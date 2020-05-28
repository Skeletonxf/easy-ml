/*!
 *
 * TODO
 */

use crate::numeric::{Numeric, NumericRef};
use crate::numeric::extra::{Real, RealRef};
use std::ops::{Add, Sub, Mul, Neg, Div};
use std::num::Wrapping;

/**
 * A trait with no methods which is implemented for all primitive types.
 *
 * Importantly this trait is not implemented for Traces, to stop the compiler
 * from trying to evaluate nested Traces of Traces as Numeric types. There is no
 * reason to create a Trace of a Trace, it won't do anything a Trace can't except
 * use more memory.
 *
 * The boilerplate implementations for primitives is performed with a macro.
 * If a primitive type is missing from this list, please open an issue to add it in.
 */
pub trait Primitive {}

macro_rules! impl_primitive {
    ($T:tt) => {
        impl Primitive for $T {}
    }
}

impl_primitive!(u8);
impl_primitive!(i8);
impl_primitive!(u16);
impl_primitive!(i16);
impl_primitive!(u32);
impl_primitive!(i32);
impl_primitive!(u64);
impl_primitive!(i64);
impl_primitive!(u128);
impl_primitive!(i128);
impl_primitive!(f32);
impl_primitive!(f64);
impl_primitive!(usize);
impl_primitive!(isize);

impl <T: Primitive> Primitive for Wrapping<T> {}

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
pub struct Trace<T: Primitive> {
    /**
     * The real number
     */
    pub number: T,
    /**
     * The first order derivative of this number.
     */
    pub derivative: T
}

impl <T: Numeric + Primitive> Trace<T> {
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
impl <T: Clone + Primitive> Clone for Trace<T> {
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
impl <T: Copy + Primitive> Copy for Trace<T> { }

/**
 * Any trace of a PartialEq type implements PartialEq
 */
impl <T: PartialEq + Primitive> PartialEq for Trace<T> {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number && self.derivative == other.derivative
    }
}

/**
 * Elementwise addition for two traces of the same type with both referenced.
 */
impl <T: Numeric + Primitive> Add for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    #[inline]
    fn add(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone() + rhs.number.clone(),
            derivative: self.derivative.clone() + rhs.derivative.clone(),
        }
    }
}

macro_rules! operator_impl_value_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
        * Elementwise operation for two traces of the same type.
        */
        impl <T: Numeric + Primitive> $op for Trace<T>
        where for<'a> &'a T: NumericRef<T> {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: Trace<T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    }
}

macro_rules! operator_impl_value_reference {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
        * Elementwise operation for two traces of the same type with the right referenced.
        */
        impl <T: Numeric + Primitive> $op<&Trace<T>> for Trace<T>
        where for<'a> &'a T: NumericRef<T> {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: &Trace<T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    }
}

macro_rules! operator_impl_reference_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
        * Elementwise operation for two traces of the same type with the left referenced.
        */
        impl <T: Numeric + Primitive> $op<Trace<T>> for &Trace<T>
        where for<'a> &'a T: NumericRef<T> {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: Trace<T>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    }
}

operator_impl_value_value!(impl Add for Trace { fn add });
operator_impl_reference_value!(impl Add for Trace { fn add });
operator_impl_value_reference!(impl Add for Trace { fn add });

/**
 * Elementwise multiplication for two referenced traces of the same type.
 */
impl <T: Numeric + Primitive> Mul for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn mul(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone() * rhs.number.clone(),
            // u'v + uv'
            derivative: (self.derivative.clone() * rhs.number.clone())
                + (self.number.clone() * rhs.derivative.clone()),
        }
    }
}

operator_impl_value_value!(impl Mul for Trace { fn mul });
operator_impl_reference_value!(impl Mul for Trace { fn mul });
operator_impl_value_reference!(impl Mul for Trace { fn mul });


/**
 * Elementwise subtraction for two referenced traces of the same type.
 */
impl <T: Numeric + Primitive> Sub for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn sub(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone() - rhs.number.clone(),
            derivative: self.derivative.clone() - rhs.derivative.clone(),
        }
    }
}

operator_impl_value_value!(impl Sub for Trace { fn sub });
operator_impl_reference_value!(impl Sub for Trace { fn sub });
operator_impl_value_reference!(impl Sub for Trace { fn sub });

/**
 * Elementwise division for two referenced traces of the same type.
 */
impl <T: Numeric + Primitive> Div for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn div(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone() / rhs.number.clone(),
            // (u'v - uv') / v^2
            derivative: (
                (
                    (self.derivative.clone() * rhs.number.clone())
                    - (self.number.clone() * rhs.derivative.clone())
                )
                / (rhs.number.clone() * rhs.number.clone())
            ),
        }
    }
}

operator_impl_value_value!(impl Div for Trace { fn div });
operator_impl_reference_value!(impl Div for Trace { fn div });
operator_impl_value_reference!(impl Div for Trace { fn div });
