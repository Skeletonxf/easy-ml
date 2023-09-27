use crate::numeric::{Numeric, NumericRef};
use crate::tensors::Tensor;
use crate::tensors::views::TensorRef;
use crate::matrices::Matrix;
use crate::matrices::views::{MatrixRef, NoInteriorMutability};
use crate::differentiation::{Primitive, Index};
use crate::differentiation::record_operations::are_same_list;
use crate::differentiation::{RecordContainer, RecordTensor, RecordMatrix};
use crate::differentiation::functions::{Addition, Subtraction, Negation, Sine, Cosine, Exponential, UnaryFunctionDerivative, FunctionDerivative};

use crate::numeric::extra::{Cos, Exp, Real, RealRef, Sin};

use std::ops::{Add, Sub, Neg};

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

macro_rules! record_matrix_operator_impl_value_value {
    (impl $op:tt for RecordMatrix { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record matrices of the same type.
         */
        impl<'a, T, S1, S2> $op<RecordMatrix<'a, T, S2>> for RecordMatrix<'a, T, S1>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: MatrixRef<(T, Index)> + NoInteriorMutability,
            S2: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: RecordMatrix<'a, T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self, rhs)
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

macro_rules! record_matrix_operator_impl_value_reference {
    (impl $op:tt for RecordMatrix { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record matrices with the right referenced.
         */
        impl<'a, T, S1, S2> $op<&RecordMatrix<'a, T, S2>> for RecordMatrix<'a, T, S1>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: MatrixRef<(T, Index)> + NoInteriorMutability,
            S2: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: &RecordMatrix<'a, T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self, rhs)
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

macro_rules! record_matrix_operator_impl_reference_value {
    (impl $op:tt for RecordMatrix { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record matrices with the left referenced.
         */
        impl<'a, T, S1, S2> $op<RecordMatrix<'a, T, S2>> for &RecordMatrix<'a, T, S1>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: MatrixRef<(T, Index)> + NoInteriorMutability,
            S2: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: RecordMatrix<'a, T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self, rhs)
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

macro_rules! record_matrix_operator_impl_reference_reference {
    (impl $op:tt for RecordMatrix { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for two record matrices with both referenced.
         */
        impl<'a, T, S1, S2> $op<&RecordMatrix<'a, T, S2>> for &RecordMatrix<'a, T, S1>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: MatrixRef<(T, Index)> + NoInteriorMutability,
            S2: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: &RecordMatrix<'a, T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self, rhs)
            }
        }
    };
}

macro_rules! record_tensor_operator_impl_value {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for a record tensor of some type.
         */
        impl<'a, T, S, const D: usize> $op for RecordTensor<'a, T, S, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self) -> Self::Output {
                $implementation::<T, S, D>(self)
            }
        }
    };
}

macro_rules! record_matrix_operator_impl_value {
    (impl $op:tt for RecordMatrix { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for a record matrix of some type.
         */
        impl<'a, T, S> $op for RecordMatrix<'a, T, S>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self) -> Self::Output {
                $implementation::<T, S>(self)
            }
        }
    };
}

macro_rules! record_tensor_operator_impl_reference {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for a referenced record tensor of some type.
         */
        impl<'a, T, S, const D: usize> $op for &RecordTensor<'a, T, S, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self) -> Self::Output {
                $implementation::<T, S, D>(self)
            }
        }
    };
}

macro_rules! record_matrix_operator_impl_reference {
    (impl $op:tt for RecordMatrix { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for a referenced record matrix of some type.
         */
        impl<'a, T, S> $op for &RecordMatrix<'a, T, S>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self) -> Self::Output {
                $implementation::<T, S>(self)
            }
        }
    };
}

macro_rules! record_real_tensor_operator_impl_value {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for a record tensor of some type.
         */
        impl<'a, T, S, const D: usize> $op for RecordTensor<'a, T, S, D>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self) -> Self::Output {
                $implementation::<T, S, D>(self)
            }
        }
    };
}

macro_rules! record_real_matrix_operator_impl_value {
    (impl $op:tt for RecordMatrix { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for a record matrix of some type.
         */
        impl<'a, T, S> $op for RecordMatrix<'a, T, S>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self) -> Self::Output {
                $implementation::<T, S>(self)
            }
        }
    };
}

macro_rules! record_real_tensor_operator_impl_reference {
    (impl $op:tt for RecordTensor { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for a referenced record tensor of some type.
         */
        impl<'a, T, S, const D: usize> $op for &RecordTensor<'a, T, S, D>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self) -> Self::Output {
                $implementation::<T, S, D>(self)
            }
        }
    };
}

macro_rules! record_real_matrix_operator_impl_reference {
    (impl $op:tt for RecordMatrix { fn $method:ident } $implementation:ident) => {
        /**
         * Operation for a referenced record matrix of some type.
         */
        impl<'a, T, S> $op for &RecordMatrix<'a, T, S>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self) -> Self::Output {
                $implementation::<T, S>(self)
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
        Addition::<T>::function,
        Addition::<T>::d_function_dx,
        Addition::<T>::d_function_dy,
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
fn record_matrix_add_allocate<'a, T, S1, S2>(
    lhs: &RecordMatrix<'a, T, S1>,
    rhs: &RecordMatrix<'a, T, S2>
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    assert!(
        are_same_list(lhs.history, rhs.history),
        "Record containers must be using the same WengertList"
    );
    lhs.binary(
        rhs,
        Addition::<T>::function,
        Addition::<T>::d_function_dx,
        Addition::<T>::d_function_dy,
    )
}

#[track_caller]
fn record_matrix_add_value_value<'a, T, S1, S2>(
    lhs: RecordMatrix<'a, T, S1>,
    rhs: RecordMatrix<'a, T, S2>
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    record_matrix_add_allocate::<T, S1, S2>(&lhs, &rhs)
}

#[track_caller]
fn record_matrix_add_value_reference<'a, T, S1, S2>(
    lhs: RecordMatrix<'a, T, S1>,
    rhs: &RecordMatrix<'a, T, S2>
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    record_matrix_add_allocate::<T, S1, S2>(&lhs, rhs)
}

#[track_caller]
fn record_matrix_add_reference_value<'a, T, S1, S2>(
    lhs: &RecordMatrix<'a, T, S1>,
    rhs: RecordMatrix<'a, T, S2>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    record_matrix_add_allocate::<T, S1, S2>(lhs, &rhs)
}

record_matrix_operator_impl_value_value!(impl Add for RecordMatrix { fn add } record_matrix_add_value_value);
record_matrix_operator_impl_value_reference!(impl Add for RecordMatrix { fn add } record_matrix_add_value_reference);
record_matrix_operator_impl_reference_value!(impl Add for RecordMatrix { fn add } record_matrix_add_reference_value);
record_matrix_operator_impl_reference_reference!(impl Add for RecordMatrix { fn add } record_matrix_add_allocate);

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
        Subtraction::<T>::function,
        Subtraction::<T>::d_function_dx,
        Subtraction::<T>::d_function_dy,
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

#[track_caller]
fn record_matrix_sub_allocate<'a, T, S1, S2>(
    lhs: &RecordMatrix<'a, T, S1>,
    rhs: &RecordMatrix<'a, T, S2>
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    assert!(
        are_same_list(lhs.history, rhs.history),
        "Record containers must be using the same WengertList"
    );
    lhs.binary(
        rhs,
        Subtraction::<T>::function,
        Subtraction::<T>::d_function_dx,
        Subtraction::<T>::d_function_dy,
    )
}

#[track_caller]
fn record_matrix_sub_value_value<'a, T, S1, S2>(
    lhs: RecordMatrix<'a, T, S1>,
    rhs: RecordMatrix<'a, T, S2>
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    record_matrix_sub_allocate::<T, S1, S2>(&lhs, &rhs)
}

#[track_caller]
fn record_matrix_sub_value_reference<'a, T, S1, S2>(
    lhs: RecordMatrix<'a, T, S1>,
    rhs: &RecordMatrix<'a, T, S2>
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    record_matrix_sub_allocate::<T, S1, S2>(&lhs, rhs)
}

#[track_caller]
fn record_matrix_sub_reference_value<'a, T, S1, S2>(
    lhs: &RecordMatrix<'a, T, S1>,
    rhs: RecordMatrix<'a, T, S2>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    record_matrix_sub_allocate::<T, S1, S2>(lhs, &rhs)
}

record_matrix_operator_impl_value_value!(impl Sub for RecordMatrix { fn sub } record_matrix_sub_value_value);
record_matrix_operator_impl_value_reference!(impl Sub for RecordMatrix { fn sub } record_matrix_sub_value_reference);
record_matrix_operator_impl_reference_value!(impl Sub for RecordMatrix { fn sub } record_matrix_sub_reference_value);
record_matrix_operator_impl_reference_reference!(impl Sub for RecordMatrix { fn sub } record_matrix_sub_allocate);

#[track_caller]
fn record_tensor_neg_value<'a, T, S, const D: usize>(
    lhs: RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(Negation::<T>::function, Negation::<T>::d_function_dx)
}

#[track_caller]
fn record_tensor_neg_reference<'a, T, S, const D: usize>(
    lhs: &RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(Negation::<T>::function, Negation::<T>::d_function_dx)
}

record_tensor_operator_impl_value!(impl Neg for RecordTensor { fn neg } record_tensor_neg_value);
record_tensor_operator_impl_reference!(impl Neg for RecordTensor { fn neg } record_tensor_neg_reference);

#[track_caller]
fn record_matrix_neg_reference<'a, T, S>(
    lhs: &RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(Negation::<T>::function, Negation::<T>::d_function_dx)
}

#[track_caller]
fn record_matrix_neg_value<'a, T, S>(
    lhs: RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(Negation::<T>::function, Negation::<T>::d_function_dx)
}

record_matrix_operator_impl_value!(impl Neg for RecordMatrix { fn neg } record_matrix_neg_value);
record_matrix_operator_impl_reference!(impl Neg for RecordMatrix { fn neg } record_matrix_neg_reference);

#[track_caller]
fn record_tensor_sin_value<'a, T, S, const D: usize>(
    lhs: RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(Sine::<T>::function, Sine::<T>::d_function_dx)
}

#[track_caller]
fn record_tensor_sin_reference<'a, T, S, const D: usize>(
    lhs: &RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(Sine::<T>::function, Sine::<T>::d_function_dx)
}

record_real_tensor_operator_impl_value!(impl Sin for RecordTensor { fn sin } record_tensor_sin_value);
record_real_tensor_operator_impl_reference!(impl Sin for RecordTensor { fn sin } record_tensor_sin_reference);

#[track_caller]
fn record_matrix_sin_reference<'a, T, S>(
    lhs: &RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(Sine::<T>::function, Sine::<T>::d_function_dx)
}

#[track_caller]
fn record_matrix_sin_value<'a, T, S>(
    lhs: RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(Sine::<T>::function, Sine::<T>::d_function_dx)
}

record_real_matrix_operator_impl_value!(impl Sin for RecordMatrix { fn sin } record_matrix_sin_value);
record_real_matrix_operator_impl_reference!(impl Sin for RecordMatrix { fn sin } record_matrix_sin_reference);

#[track_caller]
fn record_tensor_cos_value<'a, T, S, const D: usize>(
    lhs: RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(Cosine::<T>::function, Cosine::<T>::d_function_dx)
}

#[track_caller]
fn record_tensor_cos_reference<'a, T, S, const D: usize>(
    lhs: &RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(Cosine::<T>::function, Cosine::<T>::d_function_dx)
}

record_real_tensor_operator_impl_value!(impl Cos for RecordTensor { fn cos } record_tensor_cos_value);
record_real_tensor_operator_impl_reference!(impl Cos for RecordTensor { fn cos } record_tensor_cos_reference);

#[track_caller]
fn record_matrix_cos_reference<'a, T, S>(
    lhs: &RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(Cosine::<T>::function, Cosine::<T>::d_function_dx)
}

#[track_caller]
fn record_matrix_cos_value<'a, T, S>(
    lhs: RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(Cosine::<T>::function, Cosine::<T>::d_function_dx)
}

record_real_matrix_operator_impl_value!(impl Cos for RecordMatrix { fn cos } record_matrix_cos_value);
record_real_matrix_operator_impl_reference!(impl Cos for RecordMatrix { fn cos } record_matrix_cos_reference);

#[track_caller]
fn record_tensor_exp_value<'a, T, S, const D: usize>(
    lhs: RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(Exponential::<T>::function, Exponential::<T>::d_function_dx)
}

#[track_caller]
fn record_tensor_exp_reference<'a, T, S, const D: usize>(
    lhs: &RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(Exponential::<T>::function, Exponential::<T>::d_function_dx)
}

record_real_tensor_operator_impl_value!(impl Exp for RecordTensor { fn exp } record_tensor_exp_value);
record_real_tensor_operator_impl_reference!(impl Exp for RecordTensor { fn exp } record_tensor_exp_reference);

#[track_caller]
fn record_matrix_exp_reference<'a, T, S>(
    lhs: &RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(Exponential::<T>::function, Exponential::<T>::d_function_dx)
}

#[track_caller]
fn record_matrix_exp_value<'a, T, S>(
    lhs: RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(Exponential::<T>::function, Exponential::<T>::d_function_dx)
}

record_real_matrix_operator_impl_value!(impl Exp for RecordMatrix { fn exp } record_matrix_exp_value);
record_real_matrix_operator_impl_reference!(impl Exp for RecordMatrix { fn exp } record_matrix_exp_reference);
