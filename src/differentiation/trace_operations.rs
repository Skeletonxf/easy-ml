#![allow(clippy::double_parens)]
/*!
* Operator implementations for Traces
*
* These implementations are written here but Rust docs will display them on the
* [Trace] struct page.
*
* Traces of any Numeric type (provided the type also implements the operations by reference
* as described in the [numeric](super::super::numeric) module) implement all the standard
* library traits for addition, subtraction, multiplication and division, so you can
* use the normal `+ - * /` operators as you can with normal number types. As a convenience,
* these operations can also be used with a Trace on the left hand side and a the same type
* that the Trace is generic over on the right hand side, so you can do
*
* ```
* use easy_ml::differentiation::Trace;
* let x: Trace<f32> = Trace::variable(2.0);
* let y: f32 = 2.0;
* let z: Trace<f32> = x * y;
* assert_eq!(z.number, 4.0);
* ```
*
* or more succinctly
*
* ```
* use easy_ml::differentiation::Trace;
* assert_eq!((Trace::variable(2.0) * 2.0).number, 4.0);
* ```
*
* Traces of a [Real] type (provided the type also implements the operations by reference as
* described in the [numeric](super::super::numeric::extra) module) also implement
* all of those extra traits and operations. Note that to use a method defined in a trait
* you have to import the trait as well as have a type that implements it!
*/

use crate::differentiation::{Primitive, Trace};
use crate::numeric::extra::{Cos, Exp, Ln, Pi, Pow, Real, RealRef, Sin, Sqrt};
use crate::numeric::{FromUsize, Numeric, NumericRef, ZeroOne};
use std::cmp::Ordering;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};

/**
 * A trace is displayed by showing its number component.
 */
impl<T: std::fmt::Display + Primitive> std::fmt::Display for Trace<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.number)
    }
}

impl<T: Numeric + Primitive> ZeroOne for Trace<T> {
    #[inline]
    fn zero() -> Trace<T> {
        Trace::constant(T::zero())
    }
    #[inline]
    fn one() -> Trace<T> {
        Trace::constant(T::one())
    }
}

impl<T: Numeric + Primitive> FromUsize for Trace<T> {
    #[inline]
    fn from_usize(n: usize) -> Option<Trace<T>> {
        Some(Trace::constant(T::from_usize(n)?))
    }
}

/**
 * Any trace of a Cloneable type implements clone
 */
impl<T: Clone + Primitive> Clone for Trace<T> {
    #[inline]
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
impl<T: Copy + Primitive> Copy for Trace<T> {}

/**
 * Any trace of a PartialEq type implements PartialEq
 *
 * Note that as a Trace is intended to be substitutable with its
 * type T only the number parts of the trace are compared.
 * Hence the following is true
 * ```
 * use easy_ml::differentiation::Trace;
 * assert_eq!(Trace { number: 0, derivative: 1 }, Trace { number: 0, derivative: 2 })
 * ```
 */
impl<T: PartialEq + Primitive> PartialEq for Trace<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

/**
 * Any trace of a PartialOrd type implements PartialOrd
 *
 * Note that as a Trace is intended to be substitutable with its
 * type T only the number parts of the trace are compared.
 * Hence the following is true
 * ```
 * use easy_ml::differentiation::Trace;
 * assert!(Trace { number: 1, derivative: 1 } > Trace { number: 0, derivative: 2 })
 * ```
 */
impl<T: PartialOrd + Primitive> PartialOrd for Trace<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

/**
 * Any trace of a Numeric type implements Sum, which is
 * the same as adding a bunch of Trace types together.
 */
impl<T: Numeric + Primitive> Sum for Trace<T> {
    #[inline]
    fn sum<I>(mut iter: I) -> Trace<T>
    where
        I: Iterator<Item = Trace<T>>,
    {
        let mut total = Trace::<T>::zero();
        loop {
            match iter.next() {
                None => return total,
                Some(next) => {
                    total = Trace {
                        number: total.number + next.number,
                        derivative: total.derivative + next.derivative,
                    }
                }
            }
        }
    }
}

/**
 * Addition for two traces of the same type with both referenced.
 */
impl<'l, 'r, T: Numeric + Primitive> Add<&'r Trace<T>> for &'l Trace<T>
where
    for<'a> &'a T: NumericRef<T>,
{
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
         * Operation for two traces of the same type.
         */
        impl<T: Numeric + Primitive> $op for Trace<T>
        where
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: Trace<T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    };
}

macro_rules! operator_impl_value_reference {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for two traces of the same type with the right referenced.
         */
        impl<T: Numeric + Primitive> $op<&Trace<T>> for Trace<T>
        where
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: &Trace<T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

macro_rules! operator_impl_reference_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for two traces of the same type with the left referenced.
         */
        impl<T: Numeric + Primitive> $op<Trace<T>> for &Trace<T>
        where
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: Trace<T>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

operator_impl_value_value!(impl Add for Trace { fn add });
operator_impl_reference_value!(impl Add for Trace { fn add });
operator_impl_value_reference!(impl Add for Trace { fn add });

/**
 * Addition for a trace and a constant of the same type with both referenced.
 */
impl<T: Numeric + Primitive> Add<&T> for &Trace<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn add(self, rhs: &T) -> Self::Output {
        Trace {
            number: self.number.clone() + rhs.clone(),
            derivative: self.derivative.clone(),
        }
    }
}

macro_rules! trace_number_operator_impl_value_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for a trace and a constant of the same type.
         */
        impl<T: Numeric + Primitive> $op<T> for Trace<T>
        where
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    };
}

macro_rules! trace_number_operator_impl_value_reference {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for a trace and a constant of the same type with the right referenced.
         */
        impl<T: Numeric + Primitive> $op<&T> for Trace<T>
        where
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: &T) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

macro_rules! trace_number_operator_impl_reference_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for a trace and a constant of the same type with the left referenced.
         */
        impl<T: Numeric + Primitive> $op<T> for &Trace<T>
        where
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

trace_number_operator_impl_value_value!(impl Add for Trace { fn add });
trace_number_operator_impl_reference_value!(impl Add for Trace { fn add });
trace_number_operator_impl_value_reference!(impl Add for Trace { fn add });

/**
 * Multiplication for two referenced traces of the same type.
 */
impl<'l, 'r, T: Numeric + Primitive> Mul<&'r Trace<T>> for &'l Trace<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    type Output = Trace<T>;
    #[inline]
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
 * Multiplication for a trace and a constant of the same type with both referenced.
 */
impl<T: Numeric + Primitive> Mul<&T> for &Trace<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn mul(self, rhs: &T) -> Self::Output {
        Trace {
            number: self.number.clone() * rhs.clone(),
            derivative: self.derivative.clone() * rhs.clone(),
        }
    }
}

trace_number_operator_impl_value_value!(impl Mul for Trace { fn mul });
trace_number_operator_impl_reference_value!(impl Mul for Trace { fn mul });
trace_number_operator_impl_value_reference!(impl Mul for Trace { fn mul });

/**
 * Subtraction for two referenced traces of the same type.
 */
impl<'l, 'r, T: Numeric + Primitive> Sub<&'r Trace<T>> for &'l Trace<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    type Output = Trace<T>;
    #[inline]
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
 * Subtraction for a trace and a constant of the same type with both referenced.
 */
impl<T: Numeric + Primitive> Sub<&T> for &Trace<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn sub(self, rhs: &T) -> Self::Output {
        Trace {
            number: self.number.clone() - rhs.clone(),
            derivative: self.derivative.clone(),
        }
    }
}

trace_number_operator_impl_value_value!(impl Sub for Trace { fn sub });
trace_number_operator_impl_reference_value!(impl Sub for Trace { fn sub });
trace_number_operator_impl_value_reference!(impl Sub for Trace { fn sub });

/**
 * Division for two referenced traces of the same type.
 */
impl<'l, 'r, T: Numeric + Primitive> Div<&'r Trace<T>> for &'l Trace<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn div(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone() / rhs.number.clone(),
            // (u'v - uv') / v^2
            #[rustfmt::skip]
            derivative: (
                ((self.derivative.clone() * rhs.number.clone())
                - (self.number.clone() * rhs.derivative.clone()))
                / (rhs.number.clone() * rhs.number.clone())
            ),
        }
    }
}

operator_impl_value_value!(impl Div for Trace { fn div });
operator_impl_reference_value!(impl Div for Trace { fn div });
operator_impl_value_reference!(impl Div for Trace { fn div });

/**
 * Dvision for a trace and a constant of the same type with both referenced.
 */
impl<T: Numeric + Primitive> Div<&T> for &Trace<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn div(self, rhs: &T) -> Self::Output {
        Trace {
            number: self.number.clone() / rhs.clone(),
            derivative: (self.derivative.clone() * rhs.clone()) / (rhs.clone() * rhs.clone()),
        }
    }
}

trace_number_operator_impl_value_value!(impl Div for Trace { fn div });
trace_number_operator_impl_reference_value!(impl Div for Trace { fn div });
trace_number_operator_impl_value_reference!(impl Div for Trace { fn div });

/**
 * Negation for a referenced Trace of some type.
 */
impl<T: Numeric + Primitive> Neg for &Trace<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn neg(self) -> Self::Output {
        Trace::<T>::zero() - self
    }
}

/**
 * Negation for a Trace by value of some type.
 */
impl<T: Numeric + Primitive> Neg for Trace<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn neg(self) -> Self::Output {
        Trace::<T>::zero() - self
    }
}

/**
 * Sine of a Trace by reference.
 */
impl<T: Numeric + Real + Primitive> Sin for &Trace<T>
where
    for<'a> &'a T: NumericRef<T> + RealRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn sin(self) -> Self::Output {
        Trace {
            number: self.number.clone().sin(),
            // u' cos(u)
            derivative: self.derivative.clone() * self.number.clone().cos(),
        }
    }
}

macro_rules! trace_real_operator_impl_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for a trace by value.
         */
        impl<T: Numeric + Real + Primitive> $op for Trace<T>
        where
            for<'a> &'a T: NumericRef<T> + RealRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self) -> Self::Output {
                (&self).$method()
            }
        }
    };
}

trace_real_operator_impl_value!(impl Sin for Trace { fn sin });

/**
 * Cosine of a Trace by reference.
 */
impl<T: Numeric + Real + Primitive> Cos for &Trace<T>
where
    for<'a> &'a T: NumericRef<T> + RealRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn cos(self) -> Self::Output {
        Trace {
            number: self.number.clone().cos(),
            // -u' sin(u)
            derivative: -self.derivative.clone() * self.number.clone().sin(),
        }
    }
}

trace_real_operator_impl_value!(impl Cos for Trace { fn cos });

/**
 * Exponential, ie e<sup>x</sup> of a Trace by reference.
 */
impl<T: Numeric + Real + Primitive> Exp for &Trace<T>
where
    for<'a> &'a T: NumericRef<T> + RealRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn exp(self) -> Self::Output {
        Trace {
            number: self.number.clone().exp(),
            // u' exp(u)
            derivative: self.derivative.clone() * self.number.clone().exp(),
        }
    }
}

trace_real_operator_impl_value!(impl Exp for Trace { fn exp });

/**
 * Natural logarithm, ie ln(x) of a Trace by reference.
 */
impl<T: Numeric + Real + Primitive> Ln for &Trace<T>
where
    for<'a> &'a T: NumericRef<T> + RealRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn ln(self) -> Self::Output {
        Trace {
            number: self.number.clone().ln(),
            // u' / u
            derivative: self.derivative.clone() / self.number.clone(),
        }
    }
}

trace_real_operator_impl_value!(impl Ln for Trace { fn ln });

/**
 * Square root of a Trace by reference.
 */
impl<T: Numeric + Real + Primitive> Sqrt for &Trace<T>
where
    for<'a> &'a T: NumericRef<T> + RealRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn sqrt(self) -> Self::Output {
        Trace {
            number: self.number.clone().sqrt(),
            // u'/(2*sqrt(u))
            #[rustfmt::skip]
            derivative: (
                self.derivative.clone() / ((T::one() + T::one()) * self.number.clone().sqrt())
            ),
        }
    }
}

trace_real_operator_impl_value!(impl Sqrt for Trace { fn sqrt });

/**
 * Power of one Trace to another, ie self^rhs for two traces of
 * the same type with both referenced.
 */
impl<'l, 'r, T: Numeric + Real + Primitive> Pow<&'r Trace<T>> for &'l Trace<T>
where
    for<'a> &'a T: NumericRef<T> + RealRef<T>,
{
    type Output = Trace<T>;
    #[inline]
    fn pow(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone().pow(rhs.number.clone()),
            // (u' * d(u^v)/du) + (v' * d(u^v)/dv) ==
            // (u' * v * u^(v-1)) + (v' * u^v * ln(u))
            #[rustfmt::skip]
            derivative: (
                (self.derivative.clone() * rhs.number.clone()
                    * (self.number.clone().pow(rhs.number.clone() - T::one())))
                + (rhs.derivative.clone()
                    * (self.number.clone().pow(rhs.number.clone())) * self.number.clone().ln())
            ),
        }
    }
}

macro_rules! trace_real_operator_impl_value_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for two traces of the same type.
         */
        impl<T: Numeric + Real + Primitive> $op for Trace<T>
        where
            for<'a> &'a T: NumericRef<T> + RealRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: Trace<T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    };
}

macro_rules! trace_real_operator_impl_value_reference {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for two traces of the same type with the right referenced.
         */
        impl<T: Numeric + Real + Primitive> $op<&Trace<T>> for Trace<T>
        where
            for<'a> &'a T: NumericRef<T> + RealRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: &Trace<T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

macro_rules! trace_real_operator_impl_reference_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for two traces of the same type with the left referenced.
         */
        impl<T: Numeric + Real + Primitive> $op<Trace<T>> for &Trace<T>
        where
            for<'a> &'a T: NumericRef<T> + RealRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: Trace<T>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

trace_real_operator_impl_value_value!(impl Pow for Trace { fn pow });
trace_real_operator_impl_reference_value!(impl Pow for Trace { fn pow });
trace_real_operator_impl_value_reference!(impl Pow for Trace { fn pow });

/**
 * Power of a trace to a constant of the same type with both referenced.
 */
impl<T: Numeric + Real + Primitive> Pow<&T> for &Trace<T>
where
    for<'a> &'a T: NumericRef<T> + RealRef<T>,
{
    type Output = Trace<T>;
    #[allow(clippy::double_parens)]
    #[inline]
    fn pow(self, rhs: &T) -> Self::Output {
        Trace {
            number: self.number.clone().pow(rhs.clone()),
            // (u' * d(u^v)/du) == (u' * v * u^(v-1))
            #[rustfmt::skip]
            derivative: (
                (self.derivative.clone() * rhs.clone()
                * (self.number.clone().pow(rhs.clone() - T::one())))
            ),
        }
    }
}

macro_rules! trace_real_number_operator_impl_value_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for a trace and a constant of the same type.
         */
        impl<T: Numeric + Real + Primitive> $op<T> for Trace<T>
        where
            for<'a> &'a T: NumericRef<T> + RealRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    };
}

macro_rules! trace_real_number_operator_impl_value_reference {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for a trace and a constant of the same type with the right referenced.
         */
        impl<T: Numeric + Real + Primitive> $op<&T> for Trace<T>
        where
            for<'a> &'a T: NumericRef<T> + RealRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: &T) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

macro_rules! trace_real_number_operator_impl_reference_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for a trace and a constant of the same type with the left referenced.
         */
        impl<T: Numeric + Real + Primitive> $op<T> for &Trace<T>
        where
            for<'a> &'a T: NumericRef<T> + RealRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

trace_real_number_operator_impl_value_value!(impl Pow for Trace { fn pow });
trace_real_number_operator_impl_reference_value!(impl Pow for Trace { fn pow });
trace_real_number_operator_impl_value_reference!(impl Pow for Trace { fn pow });

/**
 * Power of a constant to a trace of the same type with both referenced.
 */
impl<T: Numeric + Real + Primitive> Pow<&Trace<T>> for &T
where
    for<'a> &'a T: NumericRef<T> + RealRef<T>,
{
    type Output = Trace<T>;
    #[allow(clippy::double_parens)]
    #[inline]
    fn pow(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.clone().pow(rhs.number.clone()),
            // (v' * d(u^v)/dv) == (v' * u^v * ln(u))
            #[rustfmt::skip]
            derivative:  (
                (rhs.derivative.clone()
                    * (self.clone().pow(rhs.number.clone())) * self.clone().ln())
            ),
        }
    }
}

macro_rules! real_number_trace_operator_impl_value_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for a trace and a constant of the same type.
         */
        impl<T: Numeric + Real + Primitive> $op<Trace<T>> for T
        where
            for<'a> &'a T: NumericRef<T> + RealRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: Trace<T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    };
}

macro_rules! real_number_trace_operator_impl_value_reference {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for a trace and a constant of the same type with the right referenced.
         */
        impl<T: Numeric + Real + Primitive> $op<&Trace<T>> for T
        where
            for<'a> &'a T: NumericRef<T> + RealRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: &Trace<T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

macro_rules! real_number_trace_operator_impl_reference_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
         * Operation for a trace and a constant of the same type with the left referenced.
         */
        impl<T: Numeric + Real + Primitive> $op<Trace<T>> for &T
        where
            for<'a> &'a T: NumericRef<T> + RealRef<T>,
        {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: Trace<T>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

real_number_trace_operator_impl_value_value!(impl Pow for Trace { fn pow });
real_number_trace_operator_impl_reference_value!(impl Pow for Trace { fn pow });
real_number_trace_operator_impl_value_reference!(impl Pow for Trace { fn pow });

impl<T: Numeric + Real + Primitive> Pi for Trace<T> {
    #[inline]
    fn pi() -> Trace<T> {
        Trace::constant(T::pi())
    }
}
