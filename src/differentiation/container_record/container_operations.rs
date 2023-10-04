use crate::numeric::{Numeric, NumericRef};
use crate::tensors::Tensor;
use crate::tensors::views::TensorRef;
use crate::matrices::Matrix;
use crate::matrices::views::{MatrixRef, NoInteriorMutability};
use crate::differentiation::{Primitive, WengertList, Index};
use crate::differentiation::record_operations::are_same_list;
use crate::differentiation::{RecordContainer, RecordTensor, RecordMatrix};
use crate::differentiation::functions::{Addition, Subtraction, Multiplication, Negation, NaturalLogarithm, Sine, SquareRoot, Cosine, Exponential, UnaryFunctionDerivative, FunctionDerivative};

use crate::numeric::extra::{Cos, Exp, Ln, Real, RealRef, Sin, Sqrt};

use std::ops::{Add, Sub, Mul, Neg};

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

fn record_scalar_product<'l, 'r, T, S1, S2>(
    left_iter: S1,
    right_iter: S2,
    history: Option<&WengertList<T>>,
) -> (T, Index)
where
    T: Numeric + Primitive,
    T: 'l,
    T: 'r,
    for<'a> &'a T: NumericRef<T>,
    S1: Iterator<Item = &'l (T, Index)>,
    S2: Iterator<Item = &'r (T, Index)>,
{
    match history {
        None => (
            crate::tensors::operations::scalar_product::<T, _, _>(
                left_iter.map(|(x, _)| x),
                right_iter.map(|(y, _)| y)
            ),
            0
        ),
        Some(history) => {
            let products = left_iter.zip(right_iter).map(|((x, x_index), (y, y_index))| {
                let z = Multiplication::<T>::function(x.clone(), y.clone());
                (z, history.append_binary(
                    *x_index,
                    Multiplication::<T>::d_function_dx(x.clone(), y.clone()),
                    *y_index,
                    Multiplication::<T>::d_function_dy(x.clone(), y.clone()),
                ))
            });
            products.reduce(|(x, x_index), (y, y_index)| {
                let z = Addition::<T>::function(x.clone(), y.clone());
                (z, history.append_binary(
                    x_index,
                    Addition::<T>::d_function_dx(x.clone(), y.clone()),
                    y_index,
                    Addition::<T>::d_function_dy(x, y)
                ))
            }).unwrap() // this won't be called on 0 length iterators
        }
    }
}

#[track_caller]
fn record_tensor_matrix_multiply<'a, T, S1, S2>(
    lhs: &RecordTensor<'a, T, S1, 2>,
    rhs: &RecordTensor<'a, T, S2, 2>
) -> RecordTensor<'a, T, Tensor<(T, Index), 2>, 2>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), 2>,
    S2: TensorRef<(T, Index), 2>,
{
    use crate::tensors::views::{TensorIndex, TensorView};
    use crate::tensors::indexing::TensorReferenceIterator;

    assert!(
        are_same_list(lhs.history, rhs.history),
        "Record containers must be using the same WengertList"
    );

    // TODO: Deduplicate this validation, proper error return types so can reuse messages
    let left_shape = lhs.view_shape();
    let right_shape = rhs.view_shape();
    if left_shape[1].1 != right_shape[0].1 {
        panic!(
            "Mismatched record tensors, left is {:?}, right is {:?}, * is only defined for MxN * NxL dimension lengths",
            lhs.view_shape(), rhs.view_shape()
        );
    }
    if left_shape[0].0 == right_shape[1].0 {
        panic!(
            "Matrix multiplication of record tensors with shapes left {:?} and right {:?} would \
             create duplicate dimension names as the shape {:?}. Rename one or both of the \
             dimension names in the input to prevent this. * is defined as MxN * NxL = MxL",
            left_shape,
            right_shape,
            [left_shape[0], right_shape[1]]
        )
    }

    let history = match (lhs.history, rhs.history) {
        (None, None) => None,
        (Some(history), _) => Some(history),
        (_, Some(history)) => Some(history),
    };

    // LxM * MxN -> LxN
    // [a,b,c; d,e,f] * [g,h; i,j; k,l] -> [a*g+b*i+c*k, a*h+b*j+c*l; d*g+e*i+f*k, d*h+e*j+f*l]
    // Matrix multiplication gives us another Matrix where each element [i,j] is the dot product
    // of the i'th row in the left matrix and the j'th column in the right matrix.
    let mut tensor = Tensor::empty([lhs.view_shape()[0], rhs.view_shape()[1]], (T::zero(), 0));
    for ([i, j], x) in tensor.iter_reference_mut().with_index() {
        // Select the i'th row in the left tensor to give us a vector
        let left = TensorIndex::from(&lhs, [(lhs.view_shape()[0].0, i)]);
        // Select the j'th column in the right tensor to give us a vector
        let right = TensorIndex::from(&rhs, [(rhs.view_shape()[1].0, j)]);
        // Since we checked earlier that we have MxN * NxL these two vectors have the same length.
        *x = record_scalar_product::<T, _, _>(
            TensorReferenceIterator::from(&left),
            TensorReferenceIterator::from(&right),
            history,
        )
    }
    RecordTensor::from_existing(history, TensorView::from(tensor))
}

#[track_caller]
fn record_tensor_matrix_multiply_value_value<'a, T, S1, S2>(
    lhs: RecordTensor<'a, T, S1, 2>,
    rhs: RecordTensor<'a, T, S2, 2>
) -> RecordTensor<'a, T, Tensor<(T, Index), 2>, 2>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), 2>,
    S2: TensorRef<(T, Index), 2>,
{
    record_tensor_matrix_multiply::<T, S1, S2>(&lhs, &rhs)
}

#[track_caller]
fn record_tensor_matrix_multiply_value_reference<'a, T, S1, S2>(
    lhs: RecordTensor<'a, T, S1, 2>,
    rhs: &RecordTensor<'a, T, S2, 2>
) -> RecordTensor<'a, T, Tensor<(T, Index), 2>, 2>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), 2>,
    S2: TensorRef<(T, Index), 2>,
{
    record_tensor_matrix_multiply::<T, S1, S2>(&lhs, rhs)
}

#[track_caller]
fn record_tensor_matrix_multiply_reference_value<'a, T, S1, S2>(
    lhs: &RecordTensor<'a, T, S1, 2>,
    rhs: RecordTensor<'a, T, S2, 2>
) -> RecordTensor<'a, T, Tensor<(T, Index), 2>, 2>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), 2>,
    S2: TensorRef<(T, Index), 2>,
{
    record_tensor_matrix_multiply::<T, S1, S2>(lhs, &rhs)
}

/**
 * Matrix multiplication for two record tensors with both referenced.
 */
impl<'a, T, S1, S2> Mul<&RecordTensor<'a, T, S2, 2>> for &RecordTensor<'a, T, S1, 2>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), 2>,
    S2: TensorRef<(T, Index), 2>,
{
    type Output = RecordTensor<'a, T, Tensor<(T, Index), 2>, 2>;
    #[track_caller]
    fn mul(self, rhs: &RecordTensor<'a, T, S2, 2>) -> Self::Output {
        record_tensor_matrix_multiply::<T, S1, S2>(self, rhs)
    }
}

/**
 * Matrix multiplication for two record tensors of the same type.
 */
impl<'a, T, S1, S2> Mul<RecordTensor<'a, T, S2, 2>> for RecordTensor<'a, T, S1, 2>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), 2>,
    S2: TensorRef<(T, Index), 2>,
{
    type Output = RecordTensor<'a, T, Tensor<(T, Index), 2>, 2>;
    #[track_caller]
    fn mul(self, rhs: RecordTensor<'a, T, S2, 2>) -> Self::Output {
        record_tensor_matrix_multiply_value_value::<T, S1, S2>(self, rhs)
    }
}

/**
 * Matrix multiplication for two record tensors with the right referenced.
 */
impl<'a, T, S1, S2> Mul<&RecordTensor<'a, T, S2, 2>> for RecordTensor<'a, T, S1, 2>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), 2>,
    S2: TensorRef<(T, Index), 2>,
{
    type Output = RecordTensor<'a, T, Tensor<(T, Index), 2>, 2>;
    #[track_caller]
    fn mul(self, rhs: &RecordTensor<'a, T, S2, 2>) -> Self::Output {
        record_tensor_matrix_multiply_value_reference::<T, S1, S2>(self, rhs)
    }
}

/**
 * Matrix multiplication for two record tensors with the left referenced.
 */
impl<'a, T, S1, S2> Mul<RecordTensor<'a, T, S2, 2>> for &RecordTensor<'a, T, S1, 2>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), 2>,
    S2: TensorRef<(T, Index), 2>,
{
    type Output = RecordTensor<'a, T, Tensor<(T, Index), 2>, 2>;
    #[track_caller]
    fn mul(self, rhs: RecordTensor<'a, T, S2, 2>) -> Self::Output {
        record_tensor_matrix_multiply_reference_value::<T, S1, S2>(self, rhs)
    }
}

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

#[track_caller]
fn record_tensor_ln_value<'a, T, S, const D: usize>(
    lhs: RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(NaturalLogarithm::<T>::function, NaturalLogarithm::<T>::d_function_dx)
}

#[track_caller]
fn record_tensor_ln_reference<'a, T, S, const D: usize>(
    lhs: &RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(NaturalLogarithm::<T>::function, NaturalLogarithm::<T>::d_function_dx)
}

record_real_tensor_operator_impl_value!(impl Ln for RecordTensor { fn ln } record_tensor_ln_value);
record_real_tensor_operator_impl_reference!(impl Ln for RecordTensor { fn ln } record_tensor_ln_reference);

#[track_caller]
fn record_matrix_ln_reference<'a, T, S>(
    lhs: &RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(NaturalLogarithm::<T>::function, NaturalLogarithm::<T>::d_function_dx)
}

#[track_caller]
fn record_matrix_ln_value<'a, T, S>(
    lhs: RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(NaturalLogarithm::<T>::function, NaturalLogarithm::<T>::d_function_dx)
}

record_real_matrix_operator_impl_value!(impl Ln for RecordMatrix { fn ln } record_matrix_ln_value);
record_real_matrix_operator_impl_reference!(impl Ln for RecordMatrix { fn ln } record_matrix_ln_reference);

#[track_caller]
fn record_tensor_sqrt_value<'a, T, S, const D: usize>(
    lhs: RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(SquareRoot::<T>::function, SquareRoot::<T>::d_function_dx)
}

#[track_caller]
fn record_tensor_sqrt_reference<'a, T, S, const D: usize>(
    lhs: &RecordTensor<'a, T, S, D>,
) -> RecordTensor<'a, T, Tensor<(T, Index), D>, D>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: TensorRef<(T, Index), D>,
{
    lhs.unary(SquareRoot::<T>::function, SquareRoot::<T>::d_function_dx)
}

record_real_tensor_operator_impl_value!(impl Sqrt for RecordTensor { fn sqrt } record_tensor_sqrt_value);
record_real_tensor_operator_impl_reference!(impl Sqrt for RecordTensor { fn sqrt } record_tensor_sqrt_reference);

#[track_caller]
fn record_matrix_sqrt_reference<'a, T, S>(
    lhs: &RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(SquareRoot::<T>::function, SquareRoot::<T>::d_function_dx)
}

#[track_caller]
fn record_matrix_sqrt_value<'a, T, S>(
    lhs: RecordMatrix<'a, T, S>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Real + Primitive,
    for<'t> &'t T: NumericRef<T> + RealRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    lhs.unary(SquareRoot::<T>::function, SquareRoot::<T>::d_function_dx)
}

record_real_matrix_operator_impl_value!(impl Sqrt for RecordMatrix { fn sqrt } record_matrix_sqrt_value);
record_real_matrix_operator_impl_reference!(impl Sqrt for RecordMatrix { fn sqrt } record_matrix_sqrt_reference);
