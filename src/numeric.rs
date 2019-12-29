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
use core::num::Wrapping;

/**
 * A general purpose numeric trait that defines all the behaviour numerical matrices need
 * their types to support for math operations.
 */
pub trait Numeric where
    Self:
        Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Neg<Output = Self>
        + Sum
        + PartialOrd
        + Sized
        + Clone
        + ZeroOne,
    Self: for<'a> Add<&'a Self, Output = Self>,
    // for<'a> &'a Self: Add<Self, Output = Self>,
    //for<'a, 'b> &'a Self: Add<&'b Self, Output = Self>,
    Self: for<'a> Sub<&'a Self, Output = Self>,
    Self: for<'a> Mul<&'a Self, Output = Self>,
    Self: for<'a> Div<&'a Self, Output = Self>,
{}

// FIXME: Want to express that Numeric types should also have &T operators but can't work out
// how to get &T op T and &T op &T to be specified on Numeric, T op &T is working fine.

/**
 * Anything which implements all the super traits will automatically implement this trait too.
 * This covers primitives such as f32, f64, signed integers and
 * [Wrapped unsigned integers](https://doc.rust-lang.org/std/num/struct.Wrapping.html).
 *
 * Other types such as infinite precision numbers will probably implement nearly all of these
 * anyway, but will need to add a boilerplate implementation for ZeroOne.
 */
impl <T> Numeric for T where
    T: Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Sum
    + PartialOrd
    + Sized
    + Clone
    + ZeroOne,
    // also require that operations on two &T give T
    // this allows to make no copies for things like adding
    // two f32 matrices as they can be added by reference
    for<'a, 'b> &'a T: Add<&'b T, Output = T>,
    for<'a, 'b> &'a T: Sub<&'b T, Output = T>,
    for<'a, 'b> &'a T: Mul<&'b T, Output = T>,
    for<'a, 'b> &'a T: Div<&'b T, Output = T>,
    for<'a> &'a T: Neg<Output = T>,
    // for completeness require both &T op T and T op &T
    // &T op T -> T
    for<'a> &'a T: Add<T, Output = T>,
    for<'a> &'a T: Sub<T, Output = T>,
    for<'a> &'a T: Mul<T, Output = T>,
    for<'a> &'a T: Div<T, Output = T>,
    // T op &T -> T
    for<'a> T: Add<&'a T, Output = T>,
    for<'a> T: Sub<&'a T, Output = T>,
    for<'a> T: Mul<&'a T, Output = T>,
    for<'a> T: Div<&'a T, Output = T>,
{}

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

macro_rules! zero_one_wrapping_integral {
    ($T:ty) => {
        impl ZeroOne for $T {
            #[inline]
            fn zero() -> $T { Wrapping(0) }
            #[inline]
            fn one() -> $T { Wrapping(1) }
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
zero_one_wrapping_integral!(Wrapping<u8>);
zero_one_wrapping_integral!(Wrapping<i8>);
zero_one_wrapping_integral!(Wrapping<u16>);
zero_one_wrapping_integral!(Wrapping<i16>);
zero_one_wrapping_integral!(Wrapping<u32>);
zero_one_wrapping_integral!(Wrapping<i32>);
zero_one_wrapping_integral!(Wrapping<u64>);
zero_one_wrapping_integral!(Wrapping<i64>);
zero_one_wrapping_integral!(Wrapping<u128>);
zero_one_wrapping_integral!(Wrapping<i128>);
zero_one_float!(f32);
zero_one_float!(f64);
zero_one_integral!(usize);
zero_one_integral!(isize);
