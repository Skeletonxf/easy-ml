use crate::numeric::{Numeric, NumericRef};
use crate::differentiation::Primitive;
use crate::differentiation::record_operations::are_same_list;
use crate::differentiation::{RecordContainer, RecordTensor};

use std::ops::Add;

/**
 * A record container is displayed by showing its number components.
 */
impl<'a, T, S, const D: usize> std::fmt::Display for RecordContainer<'a, T, S, D>
where
    T: std::fmt::Display + Primitive,
    S: std::fmt::Display,
    {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.numbers)
    }
}

/**
 * Any record container of a Cloneable type implements clone
 */
impl<'a, T, S, const D: usize> Clone for RecordContainer<'a, T, S, D>
where
    T: Clone + Primitive,
    S: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        RecordContainer {
            numbers: self.numbers.clone(),
            history: self.history,
            indexes: self.indexes.clone(),
        }
    }
}

macro_rules! record_tensor_operator_impl_value_value {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record tensors of the same type.
         */
        impl<'a, T, const D: usize> $op for RecordTensor<'a, T, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
        {
            type Output = RecordTensor<'a, T, D>;
            #[track_caller]
            fn $method(self, rhs: RecordTensor<'a, T, D>) -> Self::Output {
                $implementation::<T, D>(self, rhs)
            }
        }
    };
}

macro_rules! record_tensor_operator_impl_value_reference {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record tensors with the right referenced.
         */
        impl<'a, T, const D: usize> $op<&RecordTensor<'a, T, D>> for RecordTensor<'a, T, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
        {
            type Output = RecordTensor<'a, T, D>;
            #[track_caller]
            fn $method(self, rhs: &RecordTensor<'a, T, D>) -> Self::Output {
                $implementation::<T, D>(self, rhs)
            }
        }
    };
}

macro_rules! record_tensor_operator_impl_reference_value {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record tensors with the left referenced.
         */
        impl<'a, T, const D: usize> $op<RecordTensor<'a, T, D>> for &RecordTensor<'a, T, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
        {
            type Output = RecordTensor<'a, T, D>;
            #[track_caller]
            fn $method(self, rhs: RecordTensor<'a, T, D>) -> Self::Output {
                $implementation::<T, D>(self, rhs)
            }
        }
    };
}

macro_rules! record_tensor_operator_impl_reference_reference {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record tensors with both referenced.
         */
        impl<'a, T, const D: usize> $op for &RecordTensor<'a, T, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
        {
            type Output = RecordTensor<'a, T, D>;
            #[track_caller]
            fn $method(self, rhs: &RecordTensor<'a, T, D>) -> Self::Output {
                $implementation::<T, D>(self, rhs)
            }
        }
    };
}

#[track_caller]
fn record_tensor_add_assign<'a, T, const D: usize>(
    lhs: &mut RecordTensor<'a, T, D>,
    rhs: &RecordTensor<'a, T, D>
)
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    assert!(
        are_same_list(lhs.history, rhs.history),
        "Record containers must be using the same WengertList"
    );
    lhs.binary_left_assign(
        rhs,
        |x, y| x + y,
        |_x, _y| T::one(), // δ(lhs + rhs) / lhs = 1
        |_x, _y| T::one() // δ(lhs + rhs) / rhs = 1
    )
}

#[track_caller]
fn record_tensor_add_allocate<'a, T, const D: usize>(
    lhs: &RecordTensor<'a, T, D>,
    rhs: &RecordTensor<'a, T, D>
) -> RecordTensor<'a, T, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    assert!(
        are_same_list(lhs.history, rhs.history),
        "Record containers must be using the same WengertList"
    );
    lhs.binary(
        rhs,
        |x, y| x + y,
        |_x, _y| T::one(), // δ(lhs + rhs) / lhs = 1
        |_x, _y| T::one() // δ(lhs + rhs) / rhs = 1
    )
}

#[track_caller]
fn record_tensor_add_value_value<'a, T, const D: usize>(
    mut lhs: RecordTensor<'a, T, D>,
    rhs: RecordTensor<'a, T, D>
) -> RecordTensor<'a, T, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    record_tensor_add_assign::<T, D>(&mut lhs, &rhs);
    lhs
}

#[track_caller]
fn record_tensor_add_value_reference<'a, T, const D: usize>(
    mut lhs: RecordTensor<'a, T, D>,
    rhs: &RecordTensor<'a, T, D>
) -> RecordTensor<'a, T, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    record_tensor_add_assign::<T, D>(&mut lhs, rhs);
    lhs
}

#[track_caller]
fn record_tensor_add_reference_value<'a, T, const D: usize>(
    lhs: &RecordTensor<'a, T, D>,
    mut rhs: RecordTensor<'a, T, D>
) -> RecordTensor<'a, T, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
{
    record_tensor_add_assign::<T, D>(&mut rhs, lhs);
    rhs
}

record_tensor_operator_impl_value_value!(impl Add for RecordTensor { fn add } record_tensor_add_value_value);
record_tensor_operator_impl_value_reference!(impl Add for RecordTensor { fn add } record_tensor_add_value_reference);
record_tensor_operator_impl_reference_value!(impl Add for RecordTensor { fn add } record_tensor_add_reference_value);
record_tensor_operator_impl_reference_reference!(impl Add for RecordTensor { fn add } record_tensor_add_allocate);
