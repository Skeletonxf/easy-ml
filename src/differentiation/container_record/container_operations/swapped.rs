use crate::differentiation::functions::{Division, FunctionDerivative, Subtraction};
use crate::differentiation::record_operations::SwappedOperations;
use crate::differentiation::{Index, Primitive};
use crate::differentiation::{RecordMatrix, RecordTensor};
use crate::matrices::views::{MatrixRef, NoInteriorMutability};
use crate::matrices::Matrix;
use crate::numeric::{Numeric, NumericRef};
use crate::tensors::views::TensorRef;
use crate::tensors::Tensor;

impl<'a, T, S, const D: usize> SwappedOperations<T> for RecordTensor<'a, T, S, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: TensorRef<(T, Index), D>,
{
    type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;

    /**
     * Subtraction for a record tensor and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[track_caller]
    fn sub_swapped(self, lhs: T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Subtraction::<T>::function(lhs.clone(), x),
            |x| Subtraction::<T>::d_function_dy(lhs.clone(), x),
        )
    }

    /**
     * Division for a record tensor and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[track_caller]
    fn div_swapped(self, lhs: T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Division::<T>::function(lhs.clone(), x),
            |x| Division::<T>::d_function_dy(lhs.clone(), x),
        )
    }
}

impl<'a, T, S, const D: usize> SwappedOperations<&T> for RecordTensor<'a, T, S, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: TensorRef<(T, Index), D>,
{
    type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;

    /**
     * Subtraction for a record tensor and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[track_caller]
    fn sub_swapped(self, lhs: &T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Subtraction::<T>::function(lhs.clone(), x),
            |x| Subtraction::<T>::d_function_dy(lhs.clone(), x),
        )
    }

    /**
     * Division for a record tensor and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[track_caller]
    fn div_swapped(self, lhs: &T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Division::<T>::function(lhs.clone(), x),
            |x| Division::<T>::d_function_dy(lhs.clone(), x),
        )
    }
}

impl<'a, T, S, const D: usize> SwappedOperations<T> for &RecordTensor<'a, T, S, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: TensorRef<(T, Index), D>,
{
    type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;

    /**
     * Subtraction for a record tensor and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[track_caller]
    fn sub_swapped(self, lhs: T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Subtraction::<T>::function(lhs.clone(), x),
            |x| Subtraction::<T>::d_function_dy(lhs.clone(), x),
        )
    }

    /**
     * Division for a record tensor and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[track_caller]
    fn div_swapped(self, lhs: T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Division::<T>::function(lhs.clone(), x),
            |x| Division::<T>::d_function_dy(lhs.clone(), x),
        )
    }
}

impl<'a, T, S, const D: usize> SwappedOperations<&T> for &RecordTensor<'a, T, S, D>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: TensorRef<(T, Index), D>,
{
    type Output = RecordTensor<'a, T, Tensor<(T, Index), D>, D>;

    /**
     * Subtraction for a record tensor and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[track_caller]
    fn sub_swapped(self, lhs: &T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Subtraction::<T>::function(lhs.clone(), x),
            |x| Subtraction::<T>::d_function_dy(lhs.clone(), x),
        )
    }

    /**
     * Division for a record tensor and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[track_caller]
    fn div_swapped(self, lhs: &T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Division::<T>::function(lhs.clone(), x),
            |x| Division::<T>::d_function_dy(lhs.clone(), x),
        )
    }
}

impl<'a, T, S> SwappedOperations<T> for RecordMatrix<'a, T, S>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;

    /**
     * Subtraction for a record matrix and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[track_caller]
    fn sub_swapped(self, lhs: T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Subtraction::<T>::function(lhs.clone(), x),
            |x| Subtraction::<T>::d_function_dy(lhs.clone(), x),
        )
    }

    /**
     * Division for a record matrix and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[track_caller]
    fn div_swapped(self, lhs: T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Division::<T>::function(lhs.clone(), x),
            |x| Division::<T>::d_function_dy(lhs.clone(), x),
        )
    }
}

impl<'a, T, S> SwappedOperations<&T> for RecordMatrix<'a, T, S>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;

    /**
     * Subtraction for a record matrix and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[track_caller]
    fn sub_swapped(self, lhs: &T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Subtraction::<T>::function(lhs.clone(), x),
            |x| Subtraction::<T>::d_function_dy(lhs.clone(), x),
        )
    }

    /**
     * Division for a record matrix and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[track_caller]
    fn div_swapped(self, lhs: &T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Division::<T>::function(lhs.clone(), x),
            |x| Division::<T>::d_function_dy(lhs.clone(), x),
        )
    }
}

impl<'a, T, S> SwappedOperations<T> for &RecordMatrix<'a, T, S>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;

    /**
     * Subtraction for a record matrix and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[track_caller]
    fn sub_swapped(self, lhs: T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Subtraction::<T>::function(lhs.clone(), x),
            |x| Subtraction::<T>::d_function_dy(lhs.clone(), x),
        )
    }

    /**
     * Division for a record matrix and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[track_caller]
    fn div_swapped(self, lhs: T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Division::<T>::function(lhs.clone(), x),
            |x| Division::<T>::d_function_dy(lhs.clone(), x),
        )
    }
}

impl<'a, T, S> SwappedOperations<&T> for &RecordMatrix<'a, T, S>
where
    T: Numeric + Primitive,
    for<'t> &'t T: NumericRef<T>,
    S: MatrixRef<(T, Index)> + NoInteriorMutability,
{
    type Output = RecordMatrix<'a, T, Matrix<(T, Index)>>;

    /**
     * Subtraction for a record matrix and a constant, where the constant
     * is the left hand side, ie C - record.
     */
    #[track_caller]
    fn sub_swapped(self, lhs: &T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Subtraction::<T>::function(lhs.clone(), x),
            |x| Subtraction::<T>::d_function_dy(lhs.clone(), x),
        )
    }

    /**
     * Division for a record matrix and a constant, where the constant
     * is the left hand side, ie C / record.
     */
    #[track_caller]
    fn div_swapped(self, lhs: &T) -> Self::Output {
        self.unary(
            // We want with respect to y because it is the right hand side here that we
            // need the derivative for (since left is a constant).
            |x| Division::<T>::function(lhs.clone(), x),
            |x| Division::<T>::d_function_dy(lhs.clone(), x),
        )
    }
}
