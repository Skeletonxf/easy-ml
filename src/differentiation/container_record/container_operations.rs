use crate::numeric::{Numeric, NumericRef};
use crate::tensors::Tensor;
use crate::tensors::views::TensorRef;
use crate::differentiation::{Primitive, Index};
use crate::differentiation::record_operations::are_same_list;
use crate::differentiation::{RecordContainer, RecordTensor};

use std::ops::{Add, Sub};

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
        }
    }
}

macro_rules! record_tensor_operator_impl_value_value {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record tensors of the same type.
         */
        impl<'a, T, S1, S2, const D: usize> $op<RecordTensor<'a, T, S2, D>> for RecordTensor<'a, T, S1, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: TensorRef<(T, Index), D>,
            S2: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: RecordTensor<'a, T, S2, D>) -> Self::Output {
                $implementation::<T, S1, S2, D>(self, rhs)
            }
        }
    };
}

macro_rules! record_tensor_operator_impl_value_reference {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record tensors with the right referenced.
         */
        impl<'a, T, S1, S2, const D: usize> $op<&RecordTensor<'a, T, S2, D>> for RecordTensor<'a, T, S1, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: TensorRef<(T, Index), D>,
            S2: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: &RecordTensor<'a, T, S2, D>) -> Self::Output {
                $implementation::<T, S1, S2, D>(self, rhs)
            }
        }
    };
}

macro_rules! record_tensor_operator_impl_reference_value {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record tensors with the left referenced.
         */
        impl<'a, T, S1, S2, const D: usize> $op<RecordTensor<'a, T, S2, D>> for &RecordTensor<'a, T, S1, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: TensorRef<(T, Index), D>,
            S2: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: RecordTensor<'a, T, S2, D>) -> Self::Output {
                $implementation::<T, S1, S2, D>(self, rhs)
            }
        }
    };
}

macro_rules! record_tensor_operator_impl_reference_reference {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record tensors with both referenced.
         */
        impl<'a, T, S1, S2, const D: usize> $op<&RecordTensor<'a, T, S2, D>> for &RecordTensor<'a, T, S1, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: TensorRef<(T, Index), D>,
            S2: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: &RecordTensor<'a, T, S2, D>) -> Self::Output {
                $implementation::<T, S1, S2, D>(self, rhs)
            }
        }
    };
}

// We can write an add_assign variant which uses binary_left_assign instead, however
// we'd assign to a RecordTensor generic over S1, which is not always Tensor. Using Box::downcast
// almost solves this, but we can't make our inputs 'static (in fact they almost never would be).
// TODO: In a future version worth looking at adding a method to TensorRef/TensorView which allows
// for casing over the implementation type actually being a Tensor, and possibly generalise from
// Tensor to switching against a generic associated type that is the desired 'output'/'base' type.
#[track_caller]
fn record_tensor_add_allocate<'a, T, S1, S2, const D: usize>(
    lhs: &RecordTensor<'a, T, S1, D>,
    rhs: &RecordTensor<'a, T, S2, D>
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), D>,
    S2: TensorRef<(T, Index), D>,
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
fn record_tensor_add_value_value<'a, T, S1, S2, const D: usize>(
    lhs: RecordTensor<'a, T, S1, D>,
    rhs: RecordTensor<'a, T, S2, D>
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), D>,
    S2: TensorRef<(T, Index), D>,
{
    record_tensor_add_allocate::<T, S1, S2, D>(&lhs, &rhs)
}

#[track_caller]
fn record_tensor_add_value_reference<'a, T, S1, S2, const D: usize>(
    lhs: RecordTensor<'a, T, S1, D>,
    rhs: &RecordTensor<'a, T, S2, D>
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), D>,
    S2: TensorRef<(T, Index), D>,
{
    record_tensor_add_allocate::<T, S1, S2, D>(&lhs, rhs)
}

#[track_caller]
fn record_tensor_add_reference_value<'a, T, S1, S2, const D: usize>(
    lhs: &RecordTensor<'a, T, S1, D>,
    rhs: RecordTensor<'a, T, S2, D>
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), D>,
    S2: TensorRef<(T, Index), D>,
{
    record_tensor_add_allocate::<T, S1, S2, D>(lhs, &rhs)
}

record_tensor_operator_impl_value_value!(impl Add for RecordTensor { fn add } record_tensor_add_value_value);
record_tensor_operator_impl_value_reference!(impl Add for RecordTensor { fn add } record_tensor_add_value_reference);
record_tensor_operator_impl_reference_value!(impl Add for RecordTensor { fn add } record_tensor_add_reference_value);
record_tensor_operator_impl_reference_reference!(impl Add for RecordTensor { fn add } record_tensor_add_allocate);

#[track_caller]
fn record_tensor_sub_allocate<'a, T, S1, S2, const D: usize>(
    lhs: &RecordTensor<'a, T, S1, D>,
    rhs: &RecordTensor<'a, T, S2, D>
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), D>,
    S2: TensorRef<(T, Index), D>,
{
    assert!(
        are_same_list(lhs.history, rhs.history),
        "Record containers must be using the same WengertList"
    );
    lhs.binary(
        rhs,
        |x, y| x - y,
        |_x, _y| T::one(), // δ(lhs - rhs) / lhs = 1
        |_x, _y| -T::one() // δ(lhs - rhs) / rhs = -1
    )
}

#[track_caller]
fn record_tensor_sub_value_value<'a, T, S1, S2, const D: usize>(
    lhs: RecordTensor<'a, T, S1, D>,
    rhs: RecordTensor<'a, T, S2, D>
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), D>,
    S2: TensorRef<(T, Index), D>,
{
    record_tensor_sub_allocate::<T, S1, S2, D>(&lhs, &rhs)
}

#[track_caller]
fn record_tensor_sub_value_reference<'a, T, S1, S2, const D: usize>(
    lhs: RecordTensor<'a, T, S1, D>,
    rhs: &RecordTensor<'a, T, S2, D>
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), D>,
    S2: TensorRef<(T, Index), D>,
{
    record_tensor_sub_allocate::<T, S1, S2, D>(&lhs, rhs)
}

#[track_caller]
fn record_tensor_sub_reference_value<'a, T, S1, S2, const D: usize>(
    lhs: &RecordTensor<'a, T, S1, D>,
    rhs: RecordTensor<'a, T, S2, D>
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), D>,
    S2: TensorRef<(T, Index), D>,
{
    record_tensor_sub_allocate::<T, S1, S2, D>(lhs, &rhs)
}

record_tensor_operator_impl_value_value!(impl Sub for RecordTensor { fn sub } record_tensor_sub_value_value);
record_tensor_operator_impl_value_reference!(impl Sub for RecordTensor { fn sub } record_tensor_sub_value_reference);
record_tensor_operator_impl_reference_value!(impl Sub for RecordTensor { fn sub } record_tensor_sub_reference_value);
record_tensor_operator_impl_reference_reference!(impl Sub for RecordTensor { fn sub } record_tensor_sub_allocate);
