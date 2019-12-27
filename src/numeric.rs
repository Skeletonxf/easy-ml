/*!
* Numerical type definitions
*/

use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Div;
use std::ops::Neg;
use std::iter::Sum;
use std::cmp::PartialOrd;
use std::marker::Sized;

/**
 * A general purpose numeric trait that defines all the behaviour numerical matrices need
 * their types to support for math operations.
 */
pub trait Numeric: Add + Sub + Mul + Div + Neg + Sum + PartialOrd + Sized + Clone + ZeroOne {}

// TODO: Want to express that Numeric types should also have &T operators but can't work out
// the syntax for this. Once work out the syntax can remove a lot of unneccessary copies.

/**
 * Anything which implements all the super traits will automatically implement this trait too.
 * This covers primitives such as f32, f64, unsigned integers and signed integers.
 *
 * Other types such as infinite precision numbers will probably implement nearly all of these
 * anyway, but will need to add a boilerplate implementation for ZeroOne.
 */
impl<T: Add + Sub + Mul + Div + Neg + Sum + PartialOrd + Sized + Clone + ZeroOne> Numeric for T {}

/**
 * A trait defining how to obtain 0 and 1 for every implementing type.
 *
 * The boilerplate implementations for primitives is performed with a macro.
 * If a primitive type is missing from this list, please open an issue to add it in.
 */
pub trait ZeroOne: Sized {
    fn zero() -> Self;
    fn one() -> Self;
}

macro_rules! zero_one_integral {
    ($T:ty) => {
        impl ZeroOne for $T {
            #[inline]
            fn zero() -> $T { 0 }
            #[inline]
            fn one() -> $T { 1 }
        }
    };
}

macro_rules! zero_one_float {
    ($T:ty) => {
        impl ZeroOne for $T {
            #[inline]
            fn zero() -> $T { 0.0 }
            #[inline]
            fn one() -> $T { 1.0 }
        }
    };
}

zero_one_integral!(u8);
zero_one_integral!(i8);
zero_one_integral!(u16);
zero_one_integral!(i16);
zero_one_integral!(u32);
zero_one_integral!(i32);
zero_one_integral!(u64);
zero_one_integral!(i64);
zero_one_integral!(u128);
zero_one_integral!(i128);
zero_one_float!(f32);
zero_one_float!(f64);
zero_one_integral!(usize);
zero_one_integral!(isize);
