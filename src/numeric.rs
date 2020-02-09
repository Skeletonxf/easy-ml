/*!
 * Numerical type definitions.
 *
 * `Numeric` together with `where for<'a> &'a T: NumericRef<T>`
 * expresses the operations in [`NumericByValue`](./trait.NumericByValue.html) for
 * all 4 combinations of by value and by reference. [`Numeric`](./trait.Numeric.html)
 * additionally adds some additional constraints only needed by value on an implementing
 * type such as `PartialOrd`, [`ZeroOne`](./trait.ZeroOne.html) and
 * [`FromUsize`](./trait.FromUsize.html).
 */

use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Div;
use std::ops::Neg;
use std::iter::Sum;
use std::cmp::PartialOrd;
use std::marker::Sized;
use std::num::Wrapping;

/**
 * A trait defining what a numeric type is in terms of by value
 * numerical operations matrices need their types to support for
 * math operations.
 *
 * The requirements are Add, Sub, Mul, Div, Neg and Sized. Note that
 * unsigned integers do not implement Neg unless they are wrapped by
 * [Wrapping](https://doc.rust-lang.org/std/num/struct.Wrapping.html).
 */
pub trait NumericByValue<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
    + Neg<Output = Output>
    + Sized {}

/**
 * Anything which implements all the super traits will automatically implement this trait too.
 * This covers primitives such as f32, f64, signed integers and
 * [Wrapped unsigned integers](https://doc.rust-lang.org/std/num/struct.Wrapping.html).
 *
 * It will not include Matrix because Matrix does not implement Div.
 * Similarly, unwrapped unsigned integers do not implement Neg so are not included.
 */
impl <T, Rhs, Output> NumericByValue<Rhs, Output> for T where
    // Div is first here because Matrix does not implement it.
    // if Add, Sub or Mul are first the rust compiler gets stuck
    // in an infinite loop considering arbitarily nested matrix
    // types, even though any level of nested Matrix types will
    // never implement Div so shouldn't be considered for
    // implementing NumericByValue
    T: Div<Rhs, Output = Output>
    + Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Neg<Output = Output>
    + Sized {}

/**
 * The trait to define `&T op T` and `&T op &T` versions for NumericByValue
 * based off the MIT/Apache 2.0 licensed code from num-traits 0.2.10:
 *
 * **This trait is not ever used directly for users of this library. You
 * don't need to deal with it unless
 * [implementing custom numeric types](../using_custom_types/index.html)
 * and even then it will be implemented automatically.**
 *
 * - http://opensource.org/licenses/MIT
 * - https://docs.rs/num-traits/0.2.10/src/num_traits/lib.rs.html#112
 *
 * The trick is that all types implementing this trait will be references,
 * so the first constraint expresses some &T which can be operated on with
 * some right hand side type T to yield a value of type T.
 *
 * In a similar way the second constraint expresses `&T op &T -> T` operations
 */
pub trait NumericRef<T>:
    // &T op T -> T
    NumericByValue<T, T>
    // &T op &T -> T
    + for<'a> NumericByValue<&'a T, T> {}

/**
 * Anything which implements all the super traits will automatically implement this trait too.
 * This covers primitives such as `&f32`, `&f64`, ie a type like `&u8` is `NumericRef<u8>`.
 */
impl <RefT, T> NumericRef<T> for RefT where
    RefT: NumericByValue<T, T>
    + for<'a> NumericByValue<&'a T, T> {}

/**
 * A general purpose numeric trait that defines all the behaviour numerical
 * matrices need their types to support for math operations.
 *
 * This trait extends the constraints in [NumericByValue](./trait.NumericByValue.html)
 * to types which also support the operations with a right hand side type
 * by reference, and adds some additional constraints needed only
 * by value on types.
 *
 * When used together with [NumericRef](./trait.NumericRef.html) this
 * expresses all 4 by value and by reference combinations for the
 * operations using the following syntax:
 *
 * ```ignore
 *  fn function_name<T: Numeric>()
 *  where for<'a> &'a T: NumericRef<T> {
 * ```
 *
 * This pair of constraints is used nearly everywhere some numeric
 * type is needed, so although this trait does not require reference
 * type methods by itself, in practise you won't be able to call many
 * functions in this library with a numeric type that doesn't.
 */
pub trait Numeric:
    // T op T -> T
    NumericByValue
    // T op &T -> T
    + for<'a> NumericByValue<&'a Self>
    + Clone
    + ZeroOne
    + FromUsize
    + Sum
    + PartialOrd {}

/**
 * All types implemeting the operations in NumericByValue with a right hand
 * side type by reference are Numeric.
 *
 * This covers primitives such as f32, f64, signed integers and
 * [Wrapped unsigned integers](https://doc.rust-lang.org/std/num/struct.Wrapping.html).
 */
impl <T> Numeric for T where T:
    NumericByValue
    + for<'a> NumericByValue<&'a T>
    + Clone
    + ZeroOne
    + FromUsize
    + Sum
    + PartialOrd {}

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

impl <T: ZeroOne> ZeroOne for Wrapping<T> {
    #[inline]
    fn zero() -> Wrapping<T> {
        Wrapping(T::zero())
    }
    #[inline]
    fn one() -> Wrapping<T> {
        Wrapping(T::one())
    }
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

/**
 * Specifies how to obtain an instance of this numeric type
 * equal to the usize primitive. If the number is too large to
 * represent in this type, `None` should be returned instead.
 *
 * The boilerplate implementations for primitives is performed with a macro.
 * If a primitive type is missing from this list, please open an issue to add it in.
 */
pub trait FromUsize: Sized {
    fn from_usize(n: usize) -> Option<Self>;
}

impl <T: FromUsize> FromUsize for Wrapping<T> {
    fn from_usize(n: usize) -> Option<Wrapping<T>> {
        Some(Wrapping(T::from_usize(n)?))
    }
}

macro_rules! from_usize_integral {
    ($T:ty) => {
        impl FromUsize for $T {
            #[inline]
            fn from_usize(n: usize) -> Option<$T> {
                if n <= (<$T>::max_value() as usize) {
                    Some(n as $T)
                } else {
                    None
                }
            }
        }
    }
}

macro_rules! from_usize_float {
    ($T:ty) => {
        impl FromUsize for $T {
            #[inline]
            fn from_usize(n: usize) -> Option<$T> {
                Some(n as $T)
            }
        }
    }
}

from_usize_integral!(u8);
from_usize_integral!(i8);
from_usize_integral!(u16);
from_usize_integral!(i16);
from_usize_integral!(u32);
from_usize_integral!(i32);
from_usize_integral!(u64);
from_usize_integral!(i64);
from_usize_integral!(u128);
from_usize_integral!(i128);
from_usize_float!(f32);
from_usize_float!(f64);
from_usize_integral!(usize);
from_usize_integral!(isize);

/**
 * Additional traits for more complex numerical operations on real numbers.
 */
pub mod extra {

/**
 * A type which can be square rooted.
 *
 * This is implemented by `f32` and `f64` by value and by reference.
 */
pub trait Sqrt {
    type Output;
    fn sqrt(self) -> Self::Output;
}

macro_rules! sqrt_float {
    ($T:ty) => {
        impl Sqrt for $T {
            type Output = $T;
            #[inline]
            fn sqrt(self) -> Self::Output {
                self.sqrt()
            }
        }
        impl Sqrt for &$T {
            type Output = $T;
            #[inline]
            fn sqrt(self) -> Self::Output {
                self.clone().sqrt()
            }
        }
    };
}

sqrt_float!(f32);
sqrt_float!(f64);

/**
 * A type which can compute e^self.
 *
 * This is implemented by `f32` and `f64` by value and by reference.
 */
pub trait Exp {
    type Output;
    fn exp(self) -> Self::Output;
}

macro_rules! exp_float {
    ($T:ty) => {
        impl Exp for $T {
            type Output = $T;
            #[inline]
            fn exp(self) -> Self::Output {
                self.exp()
            }
        }
        impl Exp for &$T {
            type Output = $T;
            #[inline]
            fn exp(self) -> Self::Output {
                self.clone().exp()
            }
        }
    };
}

exp_float!(f32);
exp_float!(f64);


/**
 * A type which can compute self^rhs.
 *
 * This is implemented by `f32` and `f64` for all combinations of
 * by value and by reference.
 */
pub trait Pow<Rhs = Self> {
    type Output;
    fn pow(self, rhs: Rhs) -> Self::Output;
}

macro_rules! pow_float {
    ($T:ty) => {
        // T ^ T
        impl Pow<$T> for $T {
            type Output = $T;
            #[inline]
            fn pow(self, rhs: Self) -> Self::Output {
                self.powf(rhs)
            }
        }
        // T ^ &T
        impl <'a> Pow<&'a $T> for $T {
            type Output = $T;
            #[inline]
            fn pow(self, rhs: &Self) -> Self::Output {
                self.powf(rhs.clone())
            }
        }
        // &T ^ T
        impl <'a> Pow<$T> for &'a $T {
            type Output = $T;
            #[inline]
            fn pow(self, rhs: $T) -> Self::Output {
                self.powf(rhs)
            }
        }
        // &T ^ &T
        impl <'a, 'b> Pow<&'b $T> for &'a $T {
            type Output = $T;
            #[inline]
            fn pow(self, rhs: &$T) -> Self::Output {
                self.powf(rhs.clone())
            }
        }
    };
}

pow_float!(f32);
pow_float!(f64);


/**
 * A type which can represent Pi.
 */
pub trait Pi {
    fn pi() -> Self;
}

impl Pi for f32 {
    fn pi() -> f32 {
        std::f32::consts::PI
    }
}

impl Pi for f64 {
    fn pi() -> f64 {
        std::f64::consts::PI
    }
}

/**
 * A type which can compute the natural logarithm of itself: ln(self).
 *
 * This is implemented by `f32` and `f64` by value and by reference.
 */
pub trait Ln {
    type Output;
    fn ln(self) -> Self::Output;
}

macro_rules! ln_float {
    ($T:ty) => {
        impl Ln for $T {
            type Output = $T;
            #[inline]
            fn ln(self) -> Self::Output {
                self.ln()
            }
        }
        impl Ln for &$T {
            type Output = $T;
            #[inline]
            fn ln(self) -> Self::Output {
                self.clone().ln()
            }
        }
    };
}

ln_float!(f32);
ln_float!(f64);

/**
 * A type which can compute the sine of itself: sin(self)
 *
 * This is implemented by `f32` and `f64` by value and by reference.
 */
pub trait Sin {
    type Output;
    fn sin(self) -> Self::Output;
}

macro_rules! sin_float {
    ($T:ty) => {
        impl Sin for $T {
            type Output = $T;
            #[inline]
            fn sin(self) -> Self::Output {
                self.sin()
            }
        }
        impl Sin for &$T {
            type Output = $T;
            #[inline]
            fn sin(self) -> Self::Output {
                self.clone().sin()
            }
        }
    };
}

sin_float!(f32);
sin_float!(f64);


/**
 * A type which can compute the cosine of itself: cos(self)
 *
 * This is implemented by `f32` and `f64` by value and by reference.
 */
pub trait Cos {
    type Output;
    fn cos(self) -> Self::Output;
}

macro_rules! cos_float {
    ($T:ty) => {
        impl Cos for $T {
            type Output = $T;
            #[inline]
            fn cos(self) -> Self::Output {
                self.cos()
            }
        }
        impl Cos for &$T {
            type Output = $T;
            #[inline]
            fn cos(self) -> Self::Output {
                self.clone().cos()
            }
        }
    };
}

cos_float!(f32);
cos_float!(f64);

/**
 * A trait defining what a real number type is in terms of by value
 * numerical operations needed on top of operations defined by Numeric
 * for some functions.
 *
 * The requirements are Sqrt, Exp, Pow, Ln, Sin, Cos and Sized.
 */
pub trait RealByValue<Rhs = Self, Output = Self>:
    Sqrt<Output = Output>
    + Exp<Output = Output>
    + Pow<Rhs, Output = Output>
    + Ln<Output = Output>
    + Sin<Output = Output>
    + Cos<Output = Output>
    + Sized {}

/**
 * Anything which implements all the super traits will automatically implement this trait too.
 * This covers primitives such as f32 & f64
 */
impl <T, Rhs, Output> RealByValue<Rhs, Output> for T where
    T: Sqrt<Output = Output>
    + Exp<Output = Output>
    + Pow<Rhs, Output = Output>
    + Ln<Output = Output>
    + Sin<Output = Output>
    + Cos<Output = Output>
    + Sized {}

/**
 * The trait to define `&T op T` and `&T op &T` versions for RealByValue
 * based off the MIT/Apache 2.0 licensed code from num-traits 0.2.10:
 *
 * **This trait is not ever used directly for users of this library. You
 * don't need to deal with it unless
 * [implementing custom numeric types](../../using_custom_types/index.html)
 * and even then it will be implemented automatically.**
 *
 * - http://opensource.org/licenses/MIT
 * - https://docs.rs/num-traits/0.2.10/src/num_traits/lib.rs.html#112
 *
 * The trick is that all types implementing this trait will be references,
 * so the first constraint expresses some &T which can be operated on with
 * some right hand side type T to yield a value of type T.
 *
 * In a similar way the second constraint expresses `&T op &T -> T` operations
 */
pub trait RealRef<T>:
    // &T op T -> T
    RealByValue<T, T>
    // &T op &T -> T
    + for<'a> RealByValue<&'a T, T> {}

/**
 * Anything which implements all the super traits will automatically implement this trait too.
 * This covers primitives such as `&f32` & `&f64`, ie a type like `&f64` is `RealRef<&f64>`.
 */
impl <RefT, T> RealRef<T> for RefT where
    RefT: RealByValue<T, T>
    + for<'a> RealByValue<&'a T, T> {}

/**
 * A general purpose extension to the numeric trait that adds many operations needed
 * for more complex math operations.
 *
 * This trait extends the constraints in [RealByValue](./trait.RealByValue.html)
 * to types which also support the operations with a right hand side type
 * by reference, and adds some additional constraints needed only
 * by value on types.
 *
 * When used together with [RealRef](./trait.RealRef.html) this
 * expresses all 4 by value and by reference combinations for the
 * operations using the following syntax:
 *
 * ```ignore
 *  fn function_name<T: Numeric + Real>()
 *  where for<'a> &'a T: NumericRef<T> + RealRef<T> {
 * ```
 *
 * This pair of constraints is used where any real number type is needed, so although
 * this trait does not require reference type methods by itself, or re-require
 * what is in Numeric, in practise you won't be able to call many
 * functions in this library with a real type that doesn't.
 */
pub trait Real:
    // T op T -> T
    RealByValue
    // T op &T -> T
    + for<'a> RealByValue<&'a Self>
    + Pi {}

/**
 * All types implemeting the operations in RealByValue with a right hand
 * side type by reference are Real.
 *
 * This covers primitives such as f32 & f64.
 */
impl <T> Real for T where T:
    RealByValue
    + for<'a> RealByValue<&'a T>
    + Pi {}

}
