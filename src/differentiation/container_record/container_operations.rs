use crate::differentiation::functions::{
    Addition, Cosine, Division, Exponential, FunctionDerivative, Multiplication, NaturalLogarithm,
    Negation, Power, Sine, SquareRoot, Subtraction, UnaryFunctionDerivative,
};
use crate::differentiation::record_operations::are_same_list;
use crate::differentiation::{Index, Primitive, WengertList};
use crate::differentiation::{RecordContainer, RecordMatrix, RecordTensor};
use crate::matrices::views::{DataLayout, MatrixMap, MatrixRef, MatrixView, NoInteriorMutability};
use crate::matrices::{Column, Matrix, Row};
use crate::numeric::{Numeric, NumericRef};
use crate::tensors::views::{TensorMap, TensorRef, TensorView};
use crate::tensors::Tensor;

use crate::numeric::extra::{Cos, Exp, Ln, Pow, Real, RealRef, Sin, Sqrt};

use std::ops::{Add, Div, Mul, Neg, Sub};

use std::marker::PhantomData;

mod swapped;

struct MatrixRefRef<'a, T, S> {
    source: &'a S,
    _type: PhantomData<T>,
}

// # Safety
//
// Since the MatrixRef we own must implement MatrixRef correctly, so do we by delegating to it,
// as we don't introduce any interior mutability.
// TODO: Make this redundant in version 2.0 and replace with blanket impls for & and &mut versions
// of types that implement MatrixRef like we have for TensorRef
unsafe impl<'a, T, S> MatrixRef<T> for MatrixRefRef<'a, T, S>
where
    S: MatrixRef<T>,
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        self.source.try_get_reference(row, column)
    }

    fn view_rows(&self) -> Row {
        self.source.view_rows()
    }

    fn view_columns(&self) -> Column {
        self.source.view_columns()
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        self.source.get_reference_unchecked(row, column)
    }

    fn data_layout(&self) -> DataLayout {
        self.source.data_layout()
    }
}

/**
 * A record matrix is displayed by showing its number components.
 */
impl<'a, T, S> std::fmt::Display for RecordMatrix<'a, T, S>
where
    T: std::fmt::Display + Primitive,
    S: MatrixRef<(T, Index)>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            MatrixView::from(MatrixMap::from(
                MatrixRefRef {
                    source: self.numbers.source_ref(),
                    _type: PhantomData
                },
                |(x, _)| x
            ))
        )
    }
}

/**
 * A record tensor is displayed by showing its number components.
 */
impl<'a, T, S, const D: usize> std::fmt::Display for RecordTensor<'a, T, S, D>
where
    T: std::fmt::Display + Primitive,
    S: TensorRef<(T, Index), D>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            TensorView::from(TensorMap::from(self.numbers.source_ref(), |(x, _)| x))
        )
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
        impl<'a, T, S1, S2, const D: usize> $op<RecordTensor<'a, T, S2, D>>
            for RecordTensor<'a, T, S1, D>
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
        impl<'a, T, S1, S2, const D: usize> $op<&RecordTensor<'a, T, S2, D>>
            for RecordTensor<'a, T, S1, D>
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
        impl<'a, T, S1, S2, const D: usize> $op<RecordTensor<'a, T, S2, D>>
            for &RecordTensor<'a, T, S1, D>
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
        impl<'a, T, S1, S2, const D: usize> $op<&RecordTensor<'a, T, S2, D>>
            for &RecordTensor<'a, T, S1, D>
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

macro_rules! record_real_tensor_operator_impl_unary {
    (impl $op:tt for RecordTensor { fn $method:ident } $function:ident) => {
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
                self.unary($function::<T>::function, $function::<T>::d_function_dx)
            }
        }

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
                self.unary($function::<T>::function, $function::<T>::d_function_dx)
            }
        }
    };
}

macro_rules! record_real_matrix_operator_impl_unary {
    (impl $op:tt for RecordMatrix { fn $method:ident } $function:ident) => {
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
                self.unary($function::<T>::function, $function::<T>::d_function_dx)
            }
        }

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
                self.unary($function::<T>::function, $function::<T>::d_function_dx)
            }
        }
    };
}

macro_rules! record_real_tensor_operator_impl_scalar {
    (impl $op:tt for RecordTensor { fn $method:ident } $function:ident) => {
        /**
         * Operation for a record tensor and a constant of the same type. The scalar is applied
         * to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S, const D: usize> $op<T> for RecordTensor<'a, T, S, D>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a referenced record tensor and a constant of the same type. The scalar
         * is applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S, const D: usize> $op<T> for &RecordTensor<'a, T, S, D>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a record tensor and a referenced constant. The scalar is
         * applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S, const D: usize> $op<&T> for RecordTensor<'a, T, S, D>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: &T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a record tensor and a constant with both referenced. The scalar is
         * applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S, const D: usize> $op<&T> for &RecordTensor<'a, T, S, D>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: &T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }
    };
}

macro_rules! record_real_tensor_operator_impl_scalar_no_orphan_rule {
    (impl $op:tt for RecordTensor { fn $method:ident } $function:ident) => {
        /**
         * Operation for a constant and a record tensor of the same type. The scalar is applied
         * to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S, const D: usize> $op<RecordTensor<'a, T, S, D>> for T
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: RecordTensor<'a, T, S, D>) -> Self::Output {
                rhs.unary(
                    // We want with respect to y because it is the right hand side here that we
                    // need the derivative for (since left is a constant).
                    |x| $function::<T>::function(self.clone(), x),
                    |x| $function::<T>::d_function_dy(self.clone(), x),
                )
            }
        }

        /**
         * Operation for a constant and a referenced record tensor of the same type. The scalar
         * is applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S, const D: usize> $op<&RecordTensor<'a, T, S, D>> for T
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: &RecordTensor<'a, T, S, D>) -> Self::Output {
                rhs.unary(
                    // We want with respect to y because it is the right hand side here that we
                    // need the derivative for (since left is a constant).
                    |x| $function::<T>::function(self.clone(), x),
                    |x| $function::<T>::d_function_dy(self.clone(), x),
                )
            }
        }

        /**
         * Operation for a referenced constant and a record tensor of the same type. The scalar
         * is applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S, const D: usize> $op<RecordTensor<'a, T, S, D>> for &T
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: RecordTensor<'a, T, S, D>) -> Self::Output {
                rhs.unary(
                    // We want with respect to y because it is the right hand side here that we
                    // need the derivative for (since left is a constant).
                    |x| $function::<T>::function(self.clone(), x),
                    |x| $function::<T>::d_function_dy(self.clone(), x),
                )
            }
        }

        /**
         * Operation for a constant and a record tensor of the same type with both referenced. The
         * scalar is applied to all elements, this is a shorthand for
         * [unary()](RecordTensor::unary).
         */
        impl<'a, T, S, const D: usize> $op<&RecordTensor<'a, T, S, D>> for &T
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: &RecordTensor<'a, T, S, D>) -> Self::Output {
                rhs.unary(
                    // We want with respect to y because it is the right hand side here that we
                    // need the derivative for (since left is a constant).
                    |x| $function::<T>::function(self.clone(), x),
                    |x| $function::<T>::d_function_dy(self.clone(), x),
                )
            }
        }
    };
}

macro_rules! record_real_matrix_operator_impl_scalar {
    (impl $op:tt for RecordMatrix { fn $method:ident } $function:ident) => {
        /**
         * Operation for a record matrix and a constant of the same type. The scalar is applied
         * to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S> $op<T> for RecordMatrix<'a, T, S>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a referenced record matrix and a constant of the same type. The scalar
         * is applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S> $op<T> for &RecordMatrix<'a, T, S>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a record matrix and a referenced constant. The scalar is
         * applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S> $op<&T> for RecordMatrix<'a, T, S>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: &T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a record matrix and a constant with both referenced. The scalar is
         * applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S> $op<&T> for &RecordMatrix<'a, T, S>
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: &T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }
    };
}

macro_rules! record_real_matrix_operator_impl_scalar_no_orphan_rule {
    (impl $op:tt for RecordMatrix { fn $method:ident } $function:ident) => {
        /**
         * Operation for a constant and a record matrix of the same type. The scalar is applied
         * to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S> $op<RecordMatrix<'a, T, S>> for T
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: RecordMatrix<'a, T, S>) -> Self::Output {
                rhs.unary(
                    // We want with respect to y because it is the right hand side here that we
                    // need the derivative for (since left is a constant).
                    |x| $function::<T>::function(self.clone(), x),
                    |x| $function::<T>::d_function_dy(self.clone(), x),
                )
            }
        }

        /**
         * Operation for a constant and a referenced record matrix of the same type. The scalar
         * is applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S> $op<&RecordMatrix<'a, T, S>> for T
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: &RecordMatrix<'a, T, S>) -> Self::Output {
                rhs.unary(
                    // We want with respect to y because it is the right hand side here that we
                    // need the derivative for (since left is a constant).
                    |x| $function::<T>::function(self.clone(), x),
                    |x| $function::<T>::d_function_dy(self.clone(), x),
                )
            }
        }

        /**
         * Operation for a referenced constant and a record matrix of the same type. The scalar is
         * applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S> $op<RecordMatrix<'a, T, S>> for &T
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: RecordMatrix<'a, T, S>) -> Self::Output {
                rhs.unary(
                    // We want with respect to y because it is the right hand side here that we
                    // need the derivative for (since left is a constant).
                    |x| $function::<T>::function(self.clone(), x),
                    |x| $function::<T>::d_function_dy(self.clone(), x),
                )
            }
        }

        /**
         * Operation for a constant and a record matrix of the same type with both referenced.
         * The scalar is applied to all elements, this is a shorthand for
         * [unary()](RecordTensor::unary).
         */
        impl<'a, T, S> $op<&RecordMatrix<'a, T, S>> for &T
        where
            T: Numeric + Real + Primitive,
            for<'t> &'t T: NumericRef<T> + RealRef<T>,
            S: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: &RecordMatrix<'a, T, S>) -> Self::Output {
                rhs.unary(
                    // We want with respect to y because it is the right hand side here that we
                    // need the derivative for (since left is a constant).
                    |x| $function::<T>::function(self.clone(), x),
                    |x| $function::<T>::d_function_dy(self.clone(), x),
                )
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
    rhs: &RecordTensor<'a, T, S2, D>,
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
    rhs: RecordTensor<'a, T, S2, D>,
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
    rhs: &RecordTensor<'a, T, S2, D>,
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
    rhs: RecordTensor<'a, T, S2, D>,
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
    rhs: &RecordMatrix<'a, T, S2>,
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
    rhs: RecordMatrix<'a, T, S2>,
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
    rhs: &RecordMatrix<'a, T, S2>,
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
    rhs: &RecordTensor<'a, T, S2, D>,
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
    rhs: RecordTensor<'a, T, S2, D>,
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
    rhs: &RecordTensor<'a, T, S2, D>,
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
    rhs: RecordTensor<'a, T, S2, D>,
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
    rhs: &RecordMatrix<'a, T, S2>,
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
    rhs: RecordMatrix<'a, T, S2>,
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
    rhs: &RecordMatrix<'a, T, S2>,
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
                right_iter.map(|(y, _)| y),
            ),
            0,
        ),
        Some(history) => {
            let products = left_iter
                .zip(right_iter)
                .map(|((x, x_index), (y, y_index))| {
                    let z = Multiplication::<T>::function(x.clone(), y.clone());
                    (
                        z,
                        history.append_binary(
                            *x_index,
                            Multiplication::<T>::d_function_dx(x.clone(), y.clone()),
                            *y_index,
                            Multiplication::<T>::d_function_dy(x.clone(), y.clone()),
                        ),
                    )
                });
            products
                .reduce(|(x, x_index), (y, y_index)| {
                    let z = Addition::<T>::function(x.clone(), y.clone());
                    (
                        z,
                        history.append_binary(
                            x_index,
                            Addition::<T>::d_function_dx(x.clone(), y.clone()),
                            y_index,
                            Addition::<T>::d_function_dy(x, y),
                        ),
                    )
                })
                .unwrap() // this won't be called on 0 length iterators
        }
    }
}

#[track_caller]
fn record_tensor_matrix_multiply<'a, T, S1, S2>(
    lhs: &RecordTensor<'a, T, S1, 2>,
    rhs: &RecordTensor<'a, T, S2, 2>,
) -> RecordTensor<'a, T, Tensor<(T, Index), 2>, 2>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: TensorRef<(T, Index), 2>,
    S2: TensorRef<(T, Index), 2>,
{
    use crate::tensors::indexing::TensorReferenceIterator;
    use crate::tensors::views::TensorIndex;

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
    rhs: RecordTensor<'a, T, S2, 2>,
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
    rhs: &RecordTensor<'a, T, S2, 2>,
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
    rhs: RecordTensor<'a, T, S2, 2>,
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
fn record_matrix_matrix_multiply<'a, T, S1, S2>(
    lhs: &RecordMatrix<'a, T, S1>,
    rhs: &RecordMatrix<'a, T, S2>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    use crate::matrices::iterators::{ColumnReferenceIterator, RowReferenceIterator};
    // LxM * MxN -> LxN
    assert!(
        lhs.view_columns() == rhs.view_rows(),
        "Mismatched Matrices, left is {}x{}, right is {}x{}, * is only defined for MxN * NxL",
        lhs.view_rows(),
        lhs.view_columns(),
        rhs.view_rows(),
        rhs.view_columns()
    );

    let history = match (lhs.history, rhs.history) {
        (None, None) => None,
        (Some(history), _) => Some(history),
        (_, Some(history)) => Some(history),
    };

    let mut result = Matrix::empty((T::zero(), 0), (lhs.view_rows(), rhs.view_columns()));
    for ((i, j), x) in result.row_major_reference_mut_iter().with_index() {
        // Select the i'th row in the left tensor to give us a vector
        let left = RowReferenceIterator::from(lhs, i);
        // Select the j'th column in the right tensor to give us a vector
        let right = ColumnReferenceIterator::from(rhs, j);
        // Since we checked earlier that we have MxN * NxL these two vectors have the same length.
        *x = record_scalar_product::<T, _, _>(left, right, history);
    }
    RecordMatrix::from_existing(history, MatrixView::from(result))
}

#[track_caller]
fn record_matrix_matrix_multiply_value_value<'a, T, S1, S2>(
    lhs: RecordMatrix<'a, T, S1>,
    rhs: RecordMatrix<'a, T, S2>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    record_matrix_matrix_multiply::<T, S1, S2>(&lhs, &rhs)
}

#[track_caller]
fn record_matrix_matrix_multiply_value_reference<'a, T, S1, S2>(
    lhs: RecordMatrix<'a, T, S1>,
    rhs: &RecordMatrix<'a, T, S2>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    record_matrix_matrix_multiply::<T, S1, S2>(&lhs, rhs)
}

#[track_caller]
fn record_matrix_matrix_multiply_reference_value<'a, T, S1, S2>(
    lhs: &RecordMatrix<'a, T, S1>,
    rhs: RecordMatrix<'a, T, S2>,
) -> RecordMatrix<'a, T, Matrix<(T, Index)>>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    record_matrix_matrix_multiply::<T, S1, S2>(lhs, &rhs)
}

/**
 * Matrix multiplication for two record matrices with both referenced.
 */
impl<'a, T, S1, S2> Mul<&RecordMatrix<'a, T, S2>> for &RecordMatrix<'a, T, S1>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
    #[track_caller]
    fn mul(self, rhs: &RecordMatrix<'a, T, S2>) -> Self::Output {
        record_matrix_matrix_multiply::<T, S1, S2>(self, rhs)
    }
}

/**
 * Matrix multiplication for two record matrices of the same type.
 */
impl<'a, T, S1, S2> Mul<RecordMatrix<'a, T, S2>> for RecordMatrix<'a, T, S1>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
    #[track_caller]
    fn mul(self, rhs: RecordMatrix<'a, T, S2>) -> Self::Output {
        record_matrix_matrix_multiply_value_value::<T, S1, S2>(self, rhs)
    }
}

/**
 * Matrix multiplication for two record matrices with the right referenced.
 */
impl<'a, T, S1, S2> Mul<&RecordMatrix<'a, T, S2>> for RecordMatrix<'a, T, S1>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
    #[track_caller]
    fn mul(self, rhs: &RecordMatrix<'a, T, S2>) -> Self::Output {
        record_matrix_matrix_multiply_value_reference::<T, S1, S2>(self, rhs)
    }
}

/**
 * Matrix multiplication for two record matrices with the left referenced.
 */
impl<'a, T, S1, S2> Mul<RecordMatrix<'a, T, S2>> for &RecordMatrix<'a, T, S1>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S1: MatrixRef<(T, Index)> + NoInteriorMutability,
    S2: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
    #[track_caller]
    fn mul(self, rhs: RecordMatrix<'a, T, S2>) -> Self::Output {
        record_matrix_matrix_multiply_reference_value::<T, S1, S2>(self, rhs)
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

record_real_tensor_operator_impl_unary!(impl Sin for RecordTensor { fn sin } Sine);
record_real_matrix_operator_impl_unary!(impl Sin for RecordMatrix { fn sin } Sine);

record_real_tensor_operator_impl_unary!(impl Cos for RecordTensor { fn cos } Cosine);
record_real_matrix_operator_impl_unary!(impl Cos for RecordMatrix { fn cos } Cosine);

record_real_tensor_operator_impl_unary!(impl Exp for RecordTensor { fn exp } Exponential);
record_real_matrix_operator_impl_unary!(impl Exp for RecordMatrix { fn exp } Exponential);

record_real_tensor_operator_impl_unary!(impl Ln for RecordTensor { fn ln } NaturalLogarithm);
record_real_matrix_operator_impl_unary!(impl Ln for RecordMatrix { fn ln } NaturalLogarithm);

record_real_tensor_operator_impl_unary!(impl Sqrt for RecordTensor { fn sqrt } SquareRoot);
record_real_matrix_operator_impl_unary!(impl Sqrt for RecordMatrix { fn sqrt } SquareRoot);

record_real_tensor_operator_impl_scalar!(impl Pow for RecordTensor { fn pow } Power);
record_real_matrix_operator_impl_scalar!(impl Pow for RecordMatrix { fn pow } Power);
record_real_tensor_operator_impl_scalar_no_orphan_rule!(impl Pow for RecordTensor { fn pow } Power);
record_real_matrix_operator_impl_scalar_no_orphan_rule!(impl Pow for RecordMatrix { fn pow } Power);

macro_rules! record_tensor_operator_impl_scalar {
    (impl $op:tt for RecordTensor { fn $method:ident } $function:ident) => {
        /**
         * Operation for a record tensor and a constant of the same type. The scalar is applied
         * to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S1, const D: usize> $op<T> for RecordTensor<'a, T, S1, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a record tensor and a constant with the right referenced. The scalar is
         * applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S1, const D: usize> $op<&T> for RecordTensor<'a, T, S1, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: &T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a record tensor and a constant with the left referenced. The scalar is
         * applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S1, const D: usize> $op<T> for &RecordTensor<'a, T, S1, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a record tensor and a constant with both referenced. The scalar is
         * applied to all elements, this is a shorthand for [unary()](RecordTensor::unary).
         */
        impl<'a, T, S1, const D: usize> $op<&T> for &RecordTensor<'a, T, S1, D>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: TensorRef<(T, Index), D>,
        {
            type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;
            #[track_caller]
            fn $method(self, rhs: &T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }
    };
}

record_tensor_operator_impl_scalar!(impl Add for RecordTensor { fn add } Addition);
record_tensor_operator_impl_scalar!(impl Sub for RecordTensor { fn sub } Subtraction);
record_tensor_operator_impl_scalar!(impl Mul for RecordTensor { fn mul } Multiplication);
record_tensor_operator_impl_scalar!(impl Div for RecordTensor { fn div } Division);

macro_rules! record_matrix_operator_impl_scalar {
    (impl $op:tt for RecordMatrix { fn $method:ident } $function:ident) => {
        /**
         * Operation for a record matrix and a constant of the same type. The scalar is applied
         * to all elements, this is a shorthand for [unary()](RecordMatrix::unary).
         */
        impl<'a, T, S1> $op<T> for RecordMatrix<'a, T, S1>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a record matrix and a constant with the right referenced. The scalar is
         * applied to all elements, this is a shorthand for [unary()](RecordMatrix::unary).
         */
        impl<'a, T, S1> $op<&T> for RecordMatrix<'a, T, S1>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: &T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a record matrix and a constant with the left referenced. The scalar is
         * applied to all elements, this is a shorthand for [unary()](RecordMatrix::unary).
         */
        impl<'a, T, S1> $op<T> for &RecordMatrix<'a, T, S1>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }

        /**
         * Operation for a record matrix and a constant with both referenced. The scalar is applied
         * to all elements, this is a shorthand for [unary()](RecordMatrix::unary).
         */
        impl<'a, T, S1> $op<&T> for &RecordMatrix<'a, T, S1>
        where
            T: Numeric + Primitive,
            for<'t> &'t T: NumericRef<T>,
            S1: MatrixRef<(T, Index)> + NoInteriorMutability,
        {
            type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;
            #[track_caller]
            fn $method(self, rhs: &T) -> Self::Output {
                self.unary(
                    |x| $function::<T>::function(x, rhs.clone()),
                    |x| $function::<T>::d_function_dx(x, rhs.clone()),
                )
            }
        }
    };
}

record_matrix_operator_impl_scalar!(impl Add for RecordMatrix { fn add } Addition);
record_matrix_operator_impl_scalar!(impl Sub for RecordMatrix { fn sub } Subtraction);
record_matrix_operator_impl_scalar!(impl Mul for RecordMatrix { fn mul } Multiplication);
record_matrix_operator_impl_scalar!(impl Div for RecordMatrix { fn div } Division);
