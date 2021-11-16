/*!
 * Operator implementations for Records.
 *
 * These implementations are written here but Rust docs will display them on the
 * [Record](super::Record) struct page.
 *
 * Records of any Numeric type (provided the type also implements the operations by reference
 * as described in the [numeric](super::super::numeric) module) implement all the standard
 * library traits for addition, subtraction, multiplication and division, so you can
 * use the normal `+ - * /` operators as you can with normal number types. As a convenience,
 * these operations can also be used with a Record on the left hand side and a the same type
 * that the Record is generic over on the right hand side, so you can do
 *
 * ```
 * use easy_ml::differentiation::{Record, WengertList};
 * let list = WengertList::new();
 * let x: Record<f32> = Record::variable(2.0, &list);
 * let y: f32 = 2.0;
 * let z: Record<f32> = x * y;
 * assert_eq!(z.number, 4.0);
 * ```
 *
 * or more succinctly
 *
 * ```
 * use easy_ml::differentiation::{Record, WengertList};
 * assert_eq!((Record::variable(2.0, &WengertList::new()) * 2.0).number, 4.0);
 * ```
 *
 * Records of a [Real](super::super::numeric::extra::Real) type (provided the type also
 * implements the operations by reference as described in the
 * [numeric](super::super::numeric::extra) module) also implement
 * all of those extra traits and operations. Note that to use a method defined in a trait
 * you have to import the trait as well as have a type that implements it!
 */

use crate::differentiation::{Primitive, Record};
use crate::numeric::extra::{Cos, Exp, Ln, Pi, Pow, Real, RealRef, Sin, Sqrt};
use crate::numeric::{FromUsize, Numeric, NumericRef, ZeroOne};
use std::cmp::Ordering;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};

/**
 * A record is displayed by showing its number component.
 */
impl<'a, T: std::fmt::Display + Primitive> std::fmt::Display for Record<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.number)
    }
}

impl<'a, T: Numeric + Primitive> ZeroOne for Record<'a, T> {
    #[inline]
    fn zero() -> Record<'a, T> {
        Record::constant(T::zero())
    }
    #[inline]
    fn one() -> Record<'a, T> {
        Record::constant(T::one())
    }
}

impl<'a, T: Numeric + Primitive> FromUsize for Record<'a, T> {
    #[inline]
    fn from_usize(n: usize) -> Option<Record<'a, T>> {
        Some(Record::constant(T::from_usize(n)?))
    }
}

/**
 * Any record of a Cloneable type implements clone
 */
impl<'a, T: Clone + Primitive> Clone for Record<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        Record {
            number: self.number.clone(),
            history: self.history,
            index: self.index,
        }
    }
}

/**
 * Any record of a Copy type implements Copy
 */
impl<'a, T: Copy + Primitive> Copy for Record<'a, T> {}

/**
 * Compares two record's referenced WengertLists.
 *
 * If either Record is missing a reference to a WengertList then
 * this is trivially 'true', in so far as we will use the WengertList of
 * the other one.
 *
 * If both records have a WengertList, then checks that the lists are
 * the same.
 */
pub(crate) fn same_list<'a, 'b, T: Primitive>(a: &Record<'a, T>, b: &Record<'b, T>) -> bool {
    match (a.history, b.history) {
        (None, None) => true,
        (Some(_), None) => true,
        (None, Some(_)) => true,
        (Some(list_a), Some(list_b)) => std::ptr::eq(list_a, list_b),
    }
}

/**
 * Addition for two records of the same type with both referenced and
 * both using the same WengertList.
 */
impl<'a, 'l, 'r, T: Numeric + Primitive> Add<&'r Record<'a, T>> for &'l Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    #[track_caller]
    #[inline]
    fn add(self, rhs: &Record<'a, T>) -> Self::Output {
        assert!(
            same_list(self, rhs),
            "Records must be using the same WengertList"
        );
        match (self.history, rhs.history) {
            // If neither inputs have a WengertList then we don't need to record
            // the computation graph at this point because neither are inputs to
            // the overall function.
            // eg f(x, y) = ((1 + 1) * x) + (2 * (1 + y)) needs the records
            // for 2x + (2 * (1 + y)) to be stored, but we don't care about the derivatives
            // for 1 + 1, because neither were inputs to f.
            (None, None) => Record {
                number: self.number.clone() + rhs.number.clone(),
                history: None,
                index: 0,
            },
            // If only one input has a WengertList treat the other as a constant
            (Some(_), None) => self + &rhs.number,
            (None, Some(_)) => rhs + &self.number,
            (Some(history), Some(_)) => Record {
                number: self.number.clone() + rhs.number.clone(),
                history: Some(history),
                index: history.append_binary(
                    self.index,
                    // δ(self + rhs) / δself = 1
                    T::one(),
                    rhs.index,
                    // δ(self + rhs) / rhs = 1
                    T::one(),
                ),
            },
        }
    }
}

/**
 * Addition for a record and a constant of the same type with both referenced.
 */
impl<'a, T: Numeric + Primitive> Add<&T> for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    #[track_caller]
    #[inline]
    fn add(self, rhs: &T) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone() + rhs.clone(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone() + rhs.clone(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(self + C) / δself = 1
                        T::one(),
                    ),
                }
            }
        }
    }
}

macro_rules! record_operator_impl_value_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for two records of the same type.
         */
        impl<'a, T: Numeric + Primitive> $op for Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T>,
        {
            type Output = Record<'a, T>;
            #[track_caller]
            #[inline]
            fn $method(self, rhs: Record<'a, T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    };
}

macro_rules! record_operator_impl_value_reference {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for two records of the same type with the right referenced.
         */
        impl<'a, T: Numeric + Primitive> $op<&Record<'a, T>> for Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T>,
        {
            type Output = Record<'a, T>;
            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Record<'a, T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

macro_rules! record_operator_impl_reference_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for two records of the same type with the left referenced.
         */
        impl<'a, T: Numeric + Primitive> $op<Record<'a, T>> for &Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T>,
        {
            type Output = Record<'a, T>;
            #[track_caller]
            #[inline]
            fn $method(self, rhs: Record<'a, T>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

record_operator_impl_value_value!(impl Add for Record { fn add });
record_operator_impl_reference_value!(impl Add for Record { fn add });
record_operator_impl_value_reference!(impl Add for Record { fn add });

macro_rules! record_number_operator_impl_value_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for a record and a constant of the same type.
         */
        impl<'a, T: Numeric + Primitive> $op<T> for Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T>,
        {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    };
}

macro_rules! record_number_operator_impl_value_reference {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for a record and a constant of the same type with the right referenced.
         */
        impl<'a, T: Numeric + Primitive> $op<&T> for Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T>,
        {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: &T) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

macro_rules! record_number_operator_impl_reference_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for a record and a constant of the same type with the left referenced.
         */
        impl<'a, T: Numeric + Primitive> $op<T> for &Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T>,
        {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

record_number_operator_impl_value_value!(impl Add for Record { fn add });
record_number_operator_impl_reference_value!(impl Add for Record { fn add });
record_number_operator_impl_value_reference!(impl Add for Record { fn add });

/**
 * Multiplication for two records of the same type with both referenced and
 * both using the same WengertList.
 */
impl<'a, 'l, 'r, T: Numeric + Primitive> Mul<&'r Record<'a, T>> for &'l Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    #[track_caller]
    #[inline]
    fn mul(self, rhs: &Record<'a, T>) -> Self::Output {
        assert!(
            same_list(self, rhs),
            "Records must be using the same WengertList"
        );
        match (self.history, rhs.history) {
            (None, None) => Record {
                number: self.number.clone() * rhs.number.clone(),
                history: None,
                index: 0,
            },
            // If only one input has a WengertList treat the other as a constant
            (Some(_), None) => self * &rhs.number,
            (None, Some(_)) => rhs * &self.number,
            (Some(history), Some(_)) => Record {
                number: self.number.clone() * rhs.number.clone(),
                history: Some(history),
                index: history.append_binary(
                    self.index,
                    // δ(self * rhs) / δself = rhs
                    rhs.number.clone(),
                    rhs.index,
                    // δ(self * rhs) / rhs = self
                    self.number.clone(),
                ),
            },
        }
    }
}

record_operator_impl_value_value!(impl Mul for Record { fn mul });
record_operator_impl_reference_value!(impl Mul for Record { fn mul });
record_operator_impl_value_reference!(impl Mul for Record { fn mul });

/**
 * Multiplication for a record and a constant of the same type with both referenced.
 */
impl<'a, T: Numeric + Primitive> Mul<&T> for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    #[track_caller]
    #[inline]
    fn mul(self, rhs: &T) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone() * rhs.clone(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone() * rhs.clone(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(self * C) / δself = C
                        rhs.clone(),
                    ),
                }
            }
        }
    }
}

record_number_operator_impl_value_value!(impl Mul for Record { fn mul });
record_number_operator_impl_reference_value!(impl Mul for Record { fn mul });
record_number_operator_impl_value_reference!(impl Mul for Record { fn mul });

/**
 * Subtraction for two records of the same type with both referenced and
 * both using the same WengertList.
 */
impl<'a, 'l, 'r, T: Numeric + Primitive> Sub<&'r Record<'a, T>> for &'l Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    #[track_caller]
    #[inline]
    fn sub(self, rhs: &Record<'a, T>) -> Self::Output {
        assert!(
            same_list(self, rhs),
            "Records must be using the same WengertList"
        );
        match (self.history, rhs.history) {
            // If neither inputs have a WengertList then we don't need to record
            // the computation graph at this point because neither are inputs to
            // the overall function.
            // eg f(x, y) = ((1 + 1) * x) + (2 * (1 + y)) needs the records
            // for 2x + (2 * (1 + y)) to be stored, but we don't care about the derivatives
            // for 1 + 1, because neither were inputs to f.
            (None, None) => Record {
                number: self.number.clone() - rhs.number.clone(),
                history: None,
                index: 0,
            },
            // If only one input has a WengertList treat the other as a constant
            (Some(_), None) => self - &rhs.number,
            // Record::constant can't be used here as that would cause an infinite loop,
            // so use the swapped version of Sub
            (None, Some(_)) => rhs.sub_swapped(self.number.clone()),
            (Some(history), Some(_)) => Record {
                number: self.number.clone() - rhs.number.clone(),
                history: Some(history),
                index: history.append_binary(
                    self.index,
                    // δ(self - rhs) / δself = 1
                    T::one(),
                    rhs.index,
                    // δ(self - rhs) / rhs = -1
                    -T::one(),
                ),
            },
        }
    }
}

record_operator_impl_value_value!(impl Sub for Record { fn sub });
record_operator_impl_reference_value!(impl Sub for Record { fn sub });
record_operator_impl_value_reference!(impl Sub for Record { fn sub });

/**
 * Subtraction for a record and a constant of the same type with both referenced.
 */
impl<'a, T: Numeric + Primitive> Sub<&T> for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    #[inline]
    fn sub(self, rhs: &T) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone() - rhs.clone(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone() - rhs.clone(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(self - C) / δself = 1
                        T::one(),
                    ),
                }
            }
        }
    }
}

record_number_operator_impl_value_value!(impl Sub for Record { fn sub });
record_number_operator_impl_reference_value!(impl Sub for Record { fn sub });
record_number_operator_impl_value_reference!(impl Sub for Record { fn sub });

/**
 * A trait which defines subtraction and division with the arguments
 * swapped around, ie 5.sub_swapped(7) would equal 2. This trait is
 * only implemented for Records and constant operations.
 *
 * Addition and Multiplication are not included because argument order
 * doesn't matter for those operations, so you can just swap the left and
 * right and get the same result.
 *
 * Implementations for Trace are not included because you can just lift
 * a constant to a Trace with ease. While you can lift constants to Records
 * with ease too, these operations allow for the avoidance of storing the
 * constant on the WengertList which saves memory.
 */
pub trait SwappedOperations<Lhs = Self> {
    type Output;
    fn sub_swapped(self, lhs: Lhs) -> Self::Output;
    fn div_swapped(self, lhs: Lhs) -> Self::Output;
}

impl<'a, T: Numeric + Primitive> SwappedOperations<&T> for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    /**
     * Subtraction for a record and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[inline]
    fn sub_swapped(self, lhs: &T) -> Self::Output {
        match self.history {
            None => Record {
                number: lhs.clone() - self.number.clone(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: lhs.clone() - self.number.clone(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(C - self) / δself = -1
                        -T::one(),
                    ),
                }
            }
        }
    }

    /**
     * Division for a record and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[inline]
    fn div_swapped(self, lhs: &T) -> Self::Output {
        match self.history {
            None => Record {
                number: lhs.clone() / self.number.clone(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: lhs.clone() / self.number.clone(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(C / self) / δself = -(C / self^2)
                        -&lhs.clone() / (self.number.clone() * self.number.clone()),
                    ),
                }
            }
        }
    }
}

impl<'a, T: Numeric + Primitive> SwappedOperations<T> for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    /**
     * Subtraction for a record and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[inline]
    fn sub_swapped(self, lhs: T) -> Self::Output {
        self.sub_swapped(&lhs)
    }

    /**
     * Division for a record and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[inline]
    fn div_swapped(self, lhs: T) -> Self::Output {
        self.div_swapped(&lhs)
    }
}

impl<'a, T: Numeric + Primitive> SwappedOperations<T> for Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    /**
     * Subtraction for a record and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[inline]
    fn sub_swapped(self, lhs: T) -> Self::Output {
        (&self).sub_swapped(&lhs)
    }

    /**
     * Division for a record and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[inline]
    fn div_swapped(self, lhs: T) -> Self::Output {
        (&self).div_swapped(&lhs)
    }
}

impl<'a, T: Numeric + Primitive> SwappedOperations<&T> for Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    /**
     * Subtraction for a record and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[inline]
    fn sub_swapped(self, lhs: &T) -> Self::Output {
        (&self).sub_swapped(lhs)
    }

    /**
     * Division for a record and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[inline]
    fn div_swapped(self, lhs: &T) -> Self::Output {
        (&self).div_swapped(lhs)
    }
}

/**
 * Dvision for two records of the same type with both referenced and
 * both using the same WengertList.
 */
impl<'a, 'l, 'r, T: Numeric + Primitive> Div<&'r Record<'a, T>> for &'l Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    #[track_caller]
    #[inline]
    fn div(self, rhs: &Record<'a, T>) -> Self::Output {
        assert!(
            same_list(self, rhs),
            "Records must be using the same WengertList"
        );
        match (self.history, rhs.history) {
            (None, None) => Record {
                number: self.number.clone() / rhs.number.clone(),
                history: None,
                index: 0,
            },
            // If only one input has a WengertList treat the other as a constant
            (Some(_), None) => self / &rhs.number,
            // Record::constant can't be used here as that would cause an infinite loop,
            // so use the swapped version of Div
            (None, Some(_)) => rhs.div_swapped(self.number.clone()),
            (Some(history), Some(_)) => Record {
                number: self.number.clone() / rhs.number.clone(),
                history: Some(history),
                index: history.append_binary(
                    self.index,
                    // δ(self / rhs) / δself = 1 / rhs
                    T::one() / rhs.number.clone(),
                    rhs.index,
                    // δ(self / rhs) / δrhs = -(self / rhs^2)
                    -&self.number / (rhs.number.clone() * rhs.number.clone()),
                ),
            },
        }
    }
}

record_operator_impl_value_value!(impl Div for Record { fn div });
record_operator_impl_reference_value!(impl Div for Record { fn div });
record_operator_impl_value_reference!(impl Div for Record { fn div });

/**
 * Division for a record and a constant of the same type with both referenced.
 */
impl<'a, T: Numeric + Primitive> Div<&T> for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    #[track_caller]
    #[inline]
    fn div(self, rhs: &T) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone() / rhs.clone(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone() / rhs.clone(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(self / C) / δself = 1 / C
                        T::one() / rhs.clone(),
                    ),
                }
            }
        }
    }
}

record_number_operator_impl_value_value!(impl Div for Record { fn div });
record_number_operator_impl_reference_value!(impl Div for Record { fn div });
record_number_operator_impl_value_reference!(impl Div for Record { fn div });

/**
 * Negation of a record by reference.
 */
impl<'a, T: Numeric + Primitive> Neg for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    #[inline]
    fn neg(self) -> Self::Output {
        match self.history {
            None => Record {
                number: -self.number.clone(),
                history: None,
                index: 0,
            },
            Some(_) => Record::constant(T::zero()) - self,
        }
    }
}

/**
 * Negation of a record by value.
 */
impl<'a, T: Numeric + Primitive> Neg for Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    type Output = Record<'a, T>;
    #[inline]
    fn neg(self) -> Self::Output {
        match self.history {
            None => Record {
                number: -self.number,
                history: None,
                index: 0,
            },
            Some(_) => Record::constant(T::zero()) - self,
        }
    }
}

/**
 * Any record of a PartialEq type implements PartialEq
 *
 * Note that as a Record is intended to be substitutable with its
 * type T only the number parts of the record are compared.
 */
impl<'a, T: PartialEq + Primitive> PartialEq for Record<'a, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

/**
 * Any record of a PartialOrd type implements PartialOrd
 *
 * Note that as a Record is intended to be substitutable with its
 * type T only the number parts of the record are compared.
 */
impl<'a, T: PartialOrd + Primitive> PartialOrd for Record<'a, T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

/**
 * Any record of a Numeric type implements Sum, which is
 * the same as adding a bunch of Record types together.
 */
impl<'a, T: Numeric + Primitive> Sum for Record<'a, T> {
    #[track_caller]
    fn sum<I>(mut iter: I) -> Record<'a, T>
    where
        I: Iterator<Item = Record<'a, T>>,
    {
        let mut total = Record::<'a, T>::zero();
        loop {
            match iter.next() {
                None => return total,
                Some(next) => {
                    total = match (total.history, next.history) {
                        (None, None) => Record {
                            number: total.number.clone() + next.number.clone(),
                            history: None,
                            index: 0,
                        },
                        // If only one input has a WengertList treat the other as a constant
                        (Some(history), None) => {
                            Record {
                                number: total.number.clone() + next.number.clone(),
                                history: Some(history),
                                index: history.append_unary(
                                    total.index,
                                    // δ(total + next) / δtotal = 1
                                    T::one(),
                                ),
                            }
                        }
                        (None, Some(history)) => {
                            Record {
                                number: total.number.clone() + next.number.clone(),
                                history: Some(history),
                                index: history.append_unary(
                                    next.index,
                                    // δ(next + total) / δnext = 1
                                    T::one(),
                                ),
                            }
                        }
                        (Some(history), Some(_)) => {
                            assert!(
                                same_list(&total, &next),
                                "Records must be using the same WengertList"
                            );
                            Record {
                                number: total.number.clone() + next.number.clone(),
                                history: Some(history),
                                index: history.append_binary(
                                    total.index,
                                    // δ(total + next) / δtotal = 1
                                    T::one(),
                                    next.index,
                                    // δ(total + next) / δnext = 1
                                    T::one(),
                                ),
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Sine of a Record by reference.
 */
impl<'a, T: Numeric + Real + Primitive> Sin for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    type Output = Record<'a, T>;
    #[inline]
    fn sin(self) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone().sin(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone().sin(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(sin(self)) / δself = cos(self)
                        self.number.clone().cos(),
                    ),
                }
            }
        }
    }
}

macro_rules! record_real_operator_impl_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for a record by value.
         */
        impl<'a, T: Numeric + Real + Primitive> $op for Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
        {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self) -> Self::Output {
                (&self).$method()
            }
        }
    };
}

record_real_operator_impl_value!(impl Sin for Record { fn sin });

/**
 * Cosine of a Record by reference.
 */
impl<'a, T: Numeric + Real + Primitive> Cos for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    type Output = Record<'a, T>;
    #[inline]
    fn cos(self) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone().cos(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone().cos(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(cos(self)) / δself = -sin(self)
                        -self.number.clone().sin(),
                    ),
                }
            }
        }
    }
}

record_real_operator_impl_value!(impl Cos for Record { fn cos });

/**
 * Exponential, ie e<sup>x</sup> of a Record by reference.
 */
impl<'a, T: Numeric + Real + Primitive> Exp for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    type Output = Record<'a, T>;
    #[inline]
    fn exp(self) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone().exp(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone().exp(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(e^self) / δself = e^self
                        self.number.clone().exp(),
                    ),
                }
            }
        }
    }
}

record_real_operator_impl_value!(impl Exp for Record { fn exp });

/**
 * Natural logarithm, ie ln(x) of a Record by reference.
 */
impl<'a, T: Numeric + Real + Primitive> Ln for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    type Output = Record<'a, T>;
    #[inline]
    fn ln(self) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone().ln(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone().ln(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(ln(self)) / δself = 1 / self
                        T::one() / self.number.clone(),
                    ),
                }
            }
        }
    }
}

record_real_operator_impl_value!(impl Ln for Record { fn ln });

/**
 * Square root of a Record by reference.
 */
impl<'a, T: Numeric + Real + Primitive> Sqrt for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    type Output = Record<'a, T>;
    #[inline]
    fn sqrt(self) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone().sqrt(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone().sqrt(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(sqrt(self)) / δself = 1 / (2*sqrt(self))
                        T::one() / ((T::one() + T::one()) * self.number.clone().sqrt()),
                    ),
                }
            }
        }
    }
}

record_real_operator_impl_value!(impl Sqrt for Record { fn sqrt });

/**
 * Power of one Record to another, ie self^rhs for two records of
 * the same type with both referenced and both using the same WengertList.
 */
impl<'a, 'l, 'r, T: Numeric + Real + Primitive> Pow<&'r Record<'a, T>> for &'l Record<'a, T>
where
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    type Output = Record<'a, T>;
    #[inline]
    #[track_caller]
    fn pow(self, rhs: &Record<'a, T>) -> Self::Output {
        assert!(
            same_list(self, rhs),
            "Records must be using the same WengertList"
        );
        match (self.history, rhs.history) {
            (None, None) => Record {
                number: self.number.clone().pow(rhs.number.clone()),
                history: None,
                index: 0,
            },
            // If only one input has a WengertList treat the other as a constant
            (Some(_), None) => self.pow(&rhs.number),
            (None, Some(_)) => (&self.number).pow(rhs),
            (Some(history), Some(_)) => Record {
                number: self.number.clone().pow(rhs.number.clone()),
                history: Some(history),
                index: history.append_binary(
                    self.index,
                    // δ(self^rhs) / δself = rhs * self^(rhs-1)
                    rhs.number.clone() * self.number.clone().pow(rhs.number.clone() - T::one()),
                    rhs.index,
                    // δ(self^rhs) / δrhs = self^rhs * ln(self)
                    (self.number.clone().pow(rhs.number.clone())) * self.number.clone().ln(),
                ),
            },
        }
    }
}

macro_rules! record_real_operator_impl_value_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for two records of the same type.
         */
        impl<'a, T: Numeric + Real + Primitive> $op for Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
        {
            type Output = Record<'a, T>;
            #[track_caller]
            #[inline]
            fn $method(self, rhs: Record<'a, T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    };
}

macro_rules! record_real_operator_impl_value_reference {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for two records of the same type with the right referenced.
         */
        impl<'a, T: Numeric + Real + Primitive> $op<&Record<'a, T>> for Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
        {
            type Output = Record<'a, T>;
            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Record<'a, T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

macro_rules! record_real_operator_impl_reference_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for two records of the same type with the left referenced.
         */
        impl<'a, T: Numeric + Real + Primitive> $op<Record<'a, T>> for &Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
        {
            type Output = Record<'a, T>;
            #[track_caller]
            #[inline]
            fn $method(self, rhs: Record<'a, T>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

record_real_operator_impl_value_value!(impl Pow for Record { fn pow });
record_real_operator_impl_reference_value!(impl Pow for Record { fn pow });
record_real_operator_impl_value_reference!(impl Pow for Record { fn pow });

/**
 * Power of one Record to a constant of the same type with both referenced.
 */
impl<'a, T: Numeric + Real + Primitive> Pow<&T> for &Record<'a, T>
where
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    type Output = Record<'a, T>;
    #[inline]
    fn pow(self, rhs: &T) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone().pow(rhs.clone()),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone().pow(rhs.clone()),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(self^rhs) / δself = rhs * self^(rhs-1)
                        rhs.clone() * self.number.clone().pow(rhs.clone() - T::one()),
                    ),
                }
            }
        }
    }
}

/**
 * Power of a constant to a Record of the same type with both referenced.
 */
impl<'a, T: Numeric + Real + Primitive> Pow<&Record<'a, T>> for &T
where
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
{
    type Output = Record<'a, T>;
    #[inline]
    fn pow(self, rhs: &Record<'a, T>) -> Self::Output {
        match rhs.history {
            None => Record {
                number: self.clone().pow(rhs.number.clone()),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.clone().pow(rhs.number.clone()),
                    history: Some(history),
                    index: history.append_unary(
                        rhs.index,
                        // δ(self^rhs) / δrhs = self^rhs * ln(self)
                        (self.clone().pow(rhs.number.clone())) * self.clone().ln(),
                    ),
                }
            }
        }
    }
}

macro_rules! record_real_number_operator_impl_value_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for a record and a constant of the same type.
         */
        impl<'a, T: Numeric + Real + Primitive> $op<T> for Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
        {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    };
}

macro_rules! record_real_number_operator_impl_value_reference {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for a record and a constant of the same type with the right referenced.
         */
        impl<'a, T: Numeric + Real + Primitive> $op<&T> for Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
        {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: &T) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

macro_rules! record_real_number_operator_impl_reference_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for a record and a constant of the same type with the left referenced.
         */
        impl<'a, T: Numeric + Real + Primitive> $op<T> for &Record<'a, T>
        where
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
        {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

record_real_number_operator_impl_value_value!(impl Pow for Record { fn pow });
record_real_number_operator_impl_reference_value!(impl Pow for Record { fn pow });
record_real_number_operator_impl_value_reference!(impl Pow for Record { fn pow });

macro_rules! real_number_record_operator_impl_value_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for a constant and a record  of the same type.
         */
        impl<'a, T: Numeric + Real + Primitive> $op<Record<'a, T>> for T
        where
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
        {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: Record<'a, T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    };
}

macro_rules! real_number_record_operator_impl_value_reference {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for a constant and a record of the same type with the right referenced.
         */
        impl<'a, T: Numeric + Real + Primitive> $op<&Record<'a, T>> for T
        where
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
        {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: &Record<'a, T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

macro_rules! real_number_record_operator_impl_reference_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for a constant and a record of the same type with the left referenced.
         */
        impl<'a, T: Numeric + Real + Primitive> $op<Record<'a, T>> for &T
        where
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
        {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: Record<'a, T>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

real_number_record_operator_impl_value_value!(impl Pow for Record { fn pow });
real_number_record_operator_impl_reference_value!(impl Pow for Record { fn pow });
real_number_record_operator_impl_value_reference!(impl Pow for Record { fn pow });

impl<'a, T: Numeric + Real + Primitive> Pi for Record<'a, T> {
    #[inline]
    fn pi() -> Record<'a, T> {
        Record::constant(T::pi())
    }
}
