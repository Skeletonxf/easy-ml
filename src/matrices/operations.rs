/*!
 * Matrix operations.
 *
 * [Numeric](crate::numeric) type matrices implement elementwise addition and subtraction and
 * matrix multiplication.
 *
 * When doing numeric operations with matrices you should be careful to not consume a matrix by
 * accidentally using it by value. All the operations are also defined on references to matrices
 * so you should favor &x * &y style notation for matrices you intend to continue using.
 * There are also convenience operations defined for a matrix and a scalar.
 *
 * These implementations are written here but Rust docs will display them on their implemented
 * types. All 16 combinations of owned and referenced [Matrix] and [MatrixView] operations are
 * implemented. All 8 left hand side assigning addition (`+=`) and subtraction (`-=`) operations
 * are also implemented.
 *
 * Matrix multiplication is such that a matrix of dimensionality (LxM) multiplied with
 * a matrix of dimensionality (MxN) yields a new matrix of dimensionality (LxN) with each element
 * corresponding to the sum of products of the ith row in the first matrix and the jth column in
 * the second matrix.
 *
 * Operations on matrices of the wrong sizes will result in a panic. No broadcasting is performed,
 * ie you cannot multiply a (NxM) matrix by a (Nx1) column vector, you must transpose one of the
 * arguments so that the operation is valid.
 */

use crate::matrices::iterators::{
    ColumnReferenceIterator, RowMajorReferenceIterator, RowMajorReferenceMutIterator,
    RowReferenceIterator,
};
use crate::matrices::views::{MatrixMut, MatrixRef, MatrixView, NoInteriorMutability};
use crate::matrices::{Column, Matrix, Row};
use crate::numeric::{Numeric, NumericRef};

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

// TODO: Unify partial eq implementations

#[track_caller]
#[inline]
fn matrix_view_addition_iter<'l, 'r, T, S1, S2>(
    left_iter: S1,
    left_size: (Row, Column),
    right_iter: S2,
    right_size: (Row, Column),
) -> Matrix<T>
where
    T: Numeric,
    T: 'l,
    T: 'r,
    for<'a> &'a T: NumericRef<T>,
    S1: Iterator<Item = &'l T>,
    S2: Iterator<Item = &'r T>,
{
    // LxM + LxM -> LxM
    assert!(
        left_size == right_size,
        "Mismatched matrices, left is {}x{}, right is {}x{}, + is only defined for MxN + MxN",
        left_size.0,
        left_size.1,
        right_size.0,
        right_size.1
    );

    let values = left_iter.zip(right_iter).map(|(x, y)| x + y).collect();
    Matrix::from_flat_row_major(left_size, values)
}

#[track_caller]
#[inline]
fn matrix_view_subtraction_iter<'l, 'r, T, S1, S2>(
    left_iter: S1,
    left_size: (Row, Column),
    right_iter: S2,
    right_size: (Row, Column),
) -> Matrix<T>
where
    T: Numeric,
    T: 'l,
    T: 'r,
    for<'a> &'a T: NumericRef<T>,
    S1: Iterator<Item = &'l T>,
    S2: Iterator<Item = &'r T>,
{
    // LxM - LxM -> LxM
    assert!(
        left_size == right_size,
        "Mismatched matrices, left is {}x{}, right is {}x{}, + is only defined for MxN + MxN",
        left_size.0,
        left_size.1,
        right_size.0,
        right_size.1
    );

    let values = left_iter.zip(right_iter).map(|(x, y)| x - y).collect();
    Matrix::from_flat_row_major(left_size, values)
}

#[track_caller]
#[inline]
fn matrix_view_assign_addition_iter<'l, 'r, T, S1, S2>(
    left_iter: S1,
    left_size: (Row, Column),
    right_iter: S2,
    right_size: (Row, Column),
) where
    T: Numeric,
    T: 'l,
    T: 'r,
    for<'a> &'a T: NumericRef<T>,
    S1: Iterator<Item = &'l mut T>,
    S2: Iterator<Item = &'r T>,
{
    // LxM + LxM -> LxM
    assert!(
        left_size == right_size,
        "Mismatched matrices, left is {}x{}, right is {}x{}, += is only defined for MxN + MxN",
        left_size.0,
        left_size.1,
        right_size.0,
        right_size.1
    );

    for (x, y) in left_iter.zip(right_iter) {
        // Numeric doesn't define &mut T + &T so we have to clone the left hand side
        // in order to add them. For all normal number types this should be exceptionally
        // cheap since the types are all Copy anyway.
        *x = x.clone() + y;
    }
}

#[track_caller]
#[inline]
fn matrix_view_assign_subtraction_iter<'l, 'r, T, S1, S2>(
    left_iter: S1,
    left_size: (Row, Column),
    right_iter: S2,
    right_size: (Row, Column),
) where
    T: Numeric,
    T: 'l,
    T: 'r,
    for<'a> &'a T: NumericRef<T>,
    S1: Iterator<Item = &'l mut T>,
    S2: Iterator<Item = &'r T>,
{
    // LxM + LxM -> LxM
    assert!(
        left_size == right_size,
        "Mismatched matrices, left is {}x{}, right is {}x{}, -= is only defined for MxN + MxN",
        left_size.0,
        left_size.1,
        right_size.0,
        right_size.1
    );

    for (x, y) in left_iter.zip(right_iter) {
        // Numeric doesn't define &mut T + &T so we have to clone the left hand side
        // in order to add them. For all normal number types this should be exceptionally
        // cheap since the types are all Copy anyway.
        *x = x.clone() - y;
    }
}

#[track_caller]
#[inline]
fn matrix_view_multiplication<T, S1, S2>(left: &S1, right: &S2) -> Matrix<T>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S1: MatrixRef<T> + NoInteriorMutability,
    S2: MatrixRef<T> + NoInteriorMutability,
{
    use crate::tensors::operations::scalar_product;
    // LxM * MxN -> LxN
    assert!(
        left.view_columns() == right.view_rows(),
        "Mismatched Matrices, left is {}x{}, right is {}x{}, * is only defined for MxN * NxL",
        left.view_rows(),
        left.view_columns(),
        right.view_rows(),
        right.view_columns()
    );

    let mut result = Matrix::empty(T::zero(), (left.view_rows(), right.view_columns()));

    for ((i, j), x) in result.row_major_reference_mut_iter().with_index() {
        // Select the i'th row in the left tensor to give us a vector
        let left = RowReferenceIterator::from(left, i);
        // Select the j'th column in the right tensor to give us a vector
        let right = ColumnReferenceIterator::from(right, j);
        // Since we checked earlier that we have MxN * NxL these two vectors have the same length.
        *x = scalar_product::<T, _, _>(left, right);
    }
    result
}

macro_rules! matrix_view_reference_matrix_view_reference_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<&MatrixView<T, S2>> for &MatrixView<T, S1>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T> + NoInteriorMutability,
            S2: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self.source_ref(), rhs.source_ref())
            }
        }
    };
}

macro_rules! matrix_view_reference_matrix_view_reference_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<&MatrixView<T, S2>> for &MatrixView<T, S1>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T> + NoInteriorMutability,
            S2: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, _, _>(
                    RowMajorReferenceIterator::from(self.source_ref()),
                    self.size(),
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_reference_matrix_view_reference_operation_iter!(impl Add for MatrixView { fn add } matrix_view_addition_iter "Elementwise addition for two referenced matrix views");
matrix_view_reference_matrix_view_reference_operation_iter!(impl Sub for MatrixView { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for two referenced matrix views");
matrix_view_reference_matrix_view_reference_operation!(impl Mul for MatrixView { fn mul } matrix_view_multiplication "Matrix multiplication for two referenced matrix views");

macro_rules! matrix_view_assign_matrix_view_reference_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<&MatrixView<T, S2>> for MatrixView<T, S1>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixMut<T> + NoInteriorMutability,
            S2: MatrixRef<T> + NoInteriorMutability,
        {
            #[track_caller]
            #[inline]
            fn $method(&mut self, rhs: &MatrixView<T, S2>) {
                let left_size = self.size();
                $implementation::<T, _, _>(
                    RowMajorReferenceMutIterator::from(self.source_ref_mut()),
                    left_size,
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_assign_matrix_view_reference_operation_iter!(impl AddAssign for MatrixView { fn add_assign } matrix_view_assign_addition_iter "Elementwise assigning addition for two referenced matrix views");
matrix_view_assign_matrix_view_reference_operation_iter!(impl SubAssign for MatrixView { fn sub_assign } matrix_view_assign_subtraction_iter "Elementwise assigning subtraction for two referenced matrix views");

macro_rules! matrix_view_reference_matrix_view_value_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<MatrixView<T, S2>> for &MatrixView<T, S1>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T> + NoInteriorMutability,
            S2: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self.source_ref(), rhs.source_ref())
            }
        }
    };
}

macro_rules! matrix_view_reference_matrix_view_value_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<MatrixView<T, S2>> for &MatrixView<T, S1>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T> + NoInteriorMutability,
            S2: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, _, _>(
                    RowMajorReferenceIterator::from(self.source_ref()),
                    self.size(),
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_reference_matrix_view_value_operation_iter!(impl Add for MatrixView { fn add } matrix_view_addition_iter "Elementwise addition for two matrix views with one referenced");
matrix_view_reference_matrix_view_value_operation_iter!(impl Sub for MatrixView { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for two matrix views with one referenced");
matrix_view_reference_matrix_view_value_operation!(impl Mul for MatrixView { fn mul } matrix_view_multiplication "Matrix multiplication for two matrix views with one referenced");

macro_rules! matrix_view_assign_matrix_view_value_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<MatrixView<T, S2>> for MatrixView<T, S1>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixMut<T> + NoInteriorMutability,
            S2: MatrixRef<T> + NoInteriorMutability,
        {
            #[track_caller]
            #[inline]
            fn $method(&mut self, rhs: MatrixView<T, S2>) {
                let left_size = self.size();
                $implementation::<T, _, _>(
                    RowMajorReferenceMutIterator::from(self.source_ref_mut()),
                    left_size,
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_assign_matrix_view_value_operation_iter!(impl AddAssign for MatrixView { fn add_assign } matrix_view_assign_addition_iter "Elementwise assigning addition for two matrix views with one referenced");
matrix_view_assign_matrix_view_value_operation_iter!(impl SubAssign for MatrixView { fn sub_assign } matrix_view_assign_subtraction_iter "Elementwise assigning subtraction for two matrix views with one referenced");

macro_rules! matrix_view_value_matrix_view_reference_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<&MatrixView<T, S2>> for MatrixView<T, S1>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T> + NoInteriorMutability,
            S2: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self.source_ref(), rhs.source_ref())
            }
        }
    };
}

macro_rules! matrix_view_value_matrix_view_reference_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<&MatrixView<T, S2>> for MatrixView<T, S1>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T> + NoInteriorMutability,
            S2: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, _, _>(
                    RowMajorReferenceIterator::from(self.source_ref()),
                    self.size(),
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_value_matrix_view_reference_operation_iter!(impl Add for MatrixView { fn add } matrix_view_addition_iter "Elementwise addition for two matrix views with one referenced");
matrix_view_value_matrix_view_reference_operation_iter!(impl Sub for MatrixView { fn sub } matrix_view_subtraction_iter "Elementwise addition for two matrix views with one referenced");
matrix_view_value_matrix_view_reference_operation!(impl Mul for MatrixView { fn mul } matrix_view_multiplication "Matrix multiplication for two matrix views with one referenced");

macro_rules! matrix_view_value_matrix_view_value_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<MatrixView<T, S2>> for MatrixView<T, S1>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T> + NoInteriorMutability,
            S2: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self.source_ref(), rhs.source_ref())
            }
        }
    };
}

macro_rules! matrix_view_value_matrix_view_value_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S1, S2> $op<MatrixView<T, S2>> for MatrixView<T, S1>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T> + NoInteriorMutability,
            S2: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, _, _>(
                    RowMajorReferenceIterator::from(self.source_ref()),
                    self.size(),
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_value_matrix_view_value_operation_iter!(impl Add for MatrixView { fn add } matrix_view_addition_iter "Elementwise addition for two matrix views");
matrix_view_value_matrix_view_value_operation_iter!(impl Sub for MatrixView { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for two matrix views");
matrix_view_value_matrix_view_value_operation!(impl Mul for MatrixView { fn mul } matrix_view_multiplication "Matrix multiplication for two matrix views");

macro_rules! matrix_view_reference_matrix_reference_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&Matrix<T>> for &MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Matrix<T>) -> Self::Output {
                $implementation::<T, S, Matrix<T>>(self.source_ref(), rhs)
            }
        }
    };
}

macro_rules! matrix_view_reference_matrix_reference_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&Matrix<T>> for &MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Matrix<T>) -> Self::Output {
                $implementation::<T, _, _>(
                    RowMajorReferenceIterator::from(self.source_ref()),
                    self.size(),
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_reference_matrix_reference_operation_iter!(impl Add for MatrixView { fn add } matrix_view_addition_iter "Elementwise addition for a referenced matrix view and a referenced matrix");
matrix_view_reference_matrix_reference_operation_iter!(impl Sub for MatrixView { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for a referenced matrix view and a referenced matrix");
matrix_view_reference_matrix_reference_operation!(impl Mul for MatrixView { fn mul } matrix_view_multiplication "Matrix multiplication for a referenced matrix view and a referenced matrix");

macro_rules! matrix_view_assign_matrix_reference_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&Matrix<T>> for MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixMut<T> + NoInteriorMutability,
        {
            #[track_caller]
            #[inline]
            fn $method(&mut self, rhs: &Matrix<T>) {
                let left_size = self.size();
                $implementation::<T, _, _>(
                    RowMajorReferenceMutIterator::from(self.source_ref_mut()),
                    left_size,
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_assign_matrix_reference_operation_iter!(impl AddAssign for MatrixView { fn add_assign } matrix_view_assign_addition_iter "Elementwise assigning addition for a referenced matrix view and a referenced matrix");
matrix_view_assign_matrix_reference_operation_iter!(impl SubAssign for MatrixView { fn sub_assign } matrix_view_assign_subtraction_iter "Elementwise assigning subtraction for a referenced matrix view and a referenced matrix");

macro_rules! matrix_view_reference_matrix_value_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<Matrix<T>> for &MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                $implementation::<T, S, Matrix<T>>(self.source_ref(), &rhs)
            }
        }
    };
}

macro_rules! matrix_view_reference_matrix_value_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<Matrix<T>> for &MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                $implementation::<T, _, _>(
                    RowMajorReferenceIterator::from(self.source_ref()),
                    self.size(),
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_reference_matrix_value_operation_iter!(impl Add for MatrixView { fn add } matrix_view_addition_iter "Elementwise addition for a referenced matrix view and a matrix");
matrix_view_reference_matrix_value_operation_iter!(impl Sub for MatrixView { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for a referenced matrix view and a matrix");
matrix_view_reference_matrix_value_operation!(impl Mul for MatrixView { fn mul } matrix_view_multiplication "Matrix multiplication for a referenced matrix view and a matrix");

macro_rules! matrix_view_assign_matrix_value_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<Matrix<T>> for MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixMut<T> + NoInteriorMutability,
        {
            #[track_caller]
            #[inline]
            fn $method(&mut self, rhs: Matrix<T>) {
                let left_size = self.size();
                $implementation::<T, _, _>(
                    RowMajorReferenceMutIterator::from(self.source_ref_mut()),
                    left_size,
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_assign_matrix_value_operation_iter!(impl AddAssign for MatrixView { fn add_assign } matrix_view_assign_addition_iter "Elementwise assigning addition for a referenced matrix view and a matrix");
matrix_view_assign_matrix_value_operation_iter!(impl SubAssign for MatrixView { fn sub_assign } matrix_view_assign_subtraction_iter "Elementwise assigning subtraction for a referenced matrix view and a matrix");

macro_rules! matrix_view_value_matrix_reference_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&Matrix<T>> for MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Matrix<T>) -> Self::Output {
                $implementation::<T, S, Matrix<T>>(self.source_ref(), rhs)
            }
        }
    };
}

macro_rules! matrix_view_value_matrix_reference_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&Matrix<T>> for MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Matrix<T>) -> Self::Output {
                $implementation::<T, _, _>(
                    RowMajorReferenceIterator::from(self.source_ref()),
                    self.size(),
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_value_matrix_reference_operation_iter!(impl Add for MatrixView { fn add } matrix_view_addition_iter "Elementwise addition for a matrix view and a referenced matrix");
matrix_view_value_matrix_reference_operation_iter!(impl Sub for MatrixView { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for a matrix view and a referenced matrix");
matrix_view_value_matrix_reference_operation!(impl Mul for MatrixView { fn mul } matrix_view_multiplication "Matrix multiplication for a matrix view and a referenced matrix");

macro_rules! matrix_view_value_matrix_value_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<Matrix<T>> for MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                $implementation::<T, S, Matrix<T>>(self.source_ref(), &rhs)
            }
        }
    };
}

macro_rules! matrix_view_value_matrix_value_operation_iter {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<Matrix<T>> for MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                $implementation::<T, _, _>(
                    RowMajorReferenceIterator::from(self.source_ref()),
                    self.size(),
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_view_value_matrix_value_operation_iter!(impl Add for MatrixView { fn add } matrix_view_addition_iter "Elementwise addition for a matrix view and a matrix");
matrix_view_value_matrix_value_operation_iter!(impl Sub for MatrixView { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for a matrix view and a matrix");
matrix_view_value_matrix_value_operation!(impl Mul for MatrixView { fn mul } matrix_view_multiplication "Matrix multiplication for a matrix view and a matrix");

macro_rules! matrix_reference_matrix_view_reference_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&MatrixView<T, S>> for &Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S>) -> Self::Output {
                $implementation::<T, Matrix<T>, S>(self, rhs.source_ref())
            }
        }
    };
}

macro_rules! matrix_reference_matrix_view_reference_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&MatrixView<T, S>> for &Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S>) -> Self::Output {
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter(),
                    self.size(),
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_reference_matrix_view_reference_operation_iter!(impl Add for Matrix { fn add } matrix_view_addition_iter "Elementwise addition for a referenced matrix and a referenced matrix view");
matrix_reference_matrix_view_reference_operation_iter!(impl Sub for Matrix { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for a referenced matrix and a referenced matrix view");
matrix_reference_matrix_view_reference_operation!(impl Mul for Matrix { fn mul } matrix_view_multiplication "Matrix multiplication for a referenced matrix and a referenced matrix view");

macro_rules! matrix_assign_matrix_view_reference_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&MatrixView<T, S>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            #[track_caller]
            #[inline]
            fn $method(&mut self, rhs: &MatrixView<T, S>) {
                let left_size = self.size();
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter_mut(),
                    left_size,
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_assign_matrix_view_reference_operation_iter!(impl AddAssign for Matrix { fn add_assign } matrix_view_assign_addition_iter "Elementwise assigning addition for a referenced matrix and a referenced matrix view");
matrix_assign_matrix_view_reference_operation_iter!(impl SubAssign for Matrix { fn sub_assign } matrix_view_assign_subtraction_iter "Elementwise assigning subtraction for a referenced matrix and a referenced matrix view");

macro_rules! matrix_reference_matrix_view_value_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<MatrixView<T, S>> for &Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S>) -> Self::Output {
                $implementation::<T, Matrix<T>, S>(self, rhs.source_ref())
            }
        }
    };
}

macro_rules! matrix_reference_matrix_view_value_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<MatrixView<T, S>> for &Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S>) -> Self::Output {
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter(),
                    self.size(),
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_reference_matrix_view_value_operation_iter!(impl Add for Matrix { fn add } matrix_view_addition_iter "Elementwise addition for a referenced matrix and a matrix view");
matrix_reference_matrix_view_value_operation_iter!(impl Sub for Matrix { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for a referenced matrix and a matrix view");
matrix_reference_matrix_view_value_operation!(impl Mul for Matrix { fn mul } matrix_view_multiplication "Matrix multiplication for a referenced matrix and a matrix view");

macro_rules! matrix_assign_matrix_view_value_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<MatrixView<T, S>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            #[track_caller]
            #[inline]
            fn $method(&mut self, rhs: MatrixView<T, S>) {
                let left_size = self.size();
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter_mut(),
                    left_size,
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_assign_matrix_view_value_operation_iter!(impl AddAssign for Matrix { fn add_assign } matrix_view_assign_addition_iter "Elementwise assigning addition for a referenced matrix and a matrix view");
matrix_assign_matrix_view_value_operation_iter!(impl SubAssign for Matrix { fn sub_assign } matrix_view_assign_subtraction_iter "Elementwise assigning subtraction for a referenced matrix and a matrix view");

macro_rules! matrix_value_matrix_view_reference_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&MatrixView<T, S>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S>) -> Self::Output {
                $implementation::<T, Matrix<T>, S>(&self, rhs.source_ref())
            }
        }
    };
}

macro_rules! matrix_value_matrix_view_reference_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<&MatrixView<T, S>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S>) -> Self::Output {
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter(),
                    self.size(),
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_value_matrix_view_reference_operation_iter!(impl Add for Matrix { fn add } matrix_view_addition_iter "Elementwise addition for a matrix and a referenced matrix view");
matrix_value_matrix_view_reference_operation_iter!(impl Sub for Matrix { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for a matrix and a referenced matrix view");
matrix_value_matrix_view_reference_operation!(impl Mul for Matrix { fn mul } matrix_view_multiplication "Matrix multiplication for a matrix and a referenced matrix view");

macro_rules! matrix_value_matrix_view_value_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<MatrixView<T, S>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S>) -> Self::Output {
                $implementation::<T, Matrix<T>, S>(&self, rhs.source_ref())
            }
        }
    };
}

macro_rules! matrix_value_matrix_view_value_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T, S> $op<MatrixView<T, S>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T> + NoInteriorMutability,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S>) -> Self::Output {
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter(),
                    self.size(),
                    RowMajorReferenceIterator::from(rhs.source_ref()),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_value_matrix_view_value_operation_iter!(impl Add for Matrix { fn add } matrix_view_addition_iter "Elementwise addition for a matrix and a matrix view");
matrix_value_matrix_view_value_operation_iter!(impl Sub for Matrix { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for a matrix and a matrix view");
matrix_value_matrix_view_value_operation!(impl Mul for Matrix { fn mul } matrix_view_multiplication "Matrix multiplication for a matrix and a matrix view");

macro_rules! matrix_reference_matrix_reference_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<&Matrix<T>> for &Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Matrix<T>) -> Self::Output {
                $implementation::<T, Matrix<T>, Matrix<T>>(self, rhs)
            }
        }
    };
}

macro_rules! matrix_reference_matrix_reference_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<&Matrix<T>> for &Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Matrix<T>) -> Self::Output {
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter(),
                    self.size(),
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_reference_matrix_reference_operation_iter!(impl Add for Matrix { fn add } matrix_view_addition_iter "Elementwise addition for two referenced matrices");
matrix_reference_matrix_reference_operation_iter!(impl Sub for Matrix { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for two referenced matrices");
matrix_reference_matrix_reference_operation!(impl Mul for Matrix { fn mul } matrix_view_multiplication "Matrix multiplication for two referenced matrices");

macro_rules! matrix_assign_matrix_reference_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<&Matrix<T>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            #[track_caller]
            #[inline]
            fn $method(&mut self, rhs: &Matrix<T>) {
                let left_size = self.size();
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter_mut(),
                    left_size,
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_assign_matrix_reference_operation_iter!(impl AddAssign for Matrix { fn add_assign } matrix_view_assign_addition_iter "Elementwise assigning addition for two referenced matrices");
matrix_assign_matrix_reference_operation_iter!(impl SubAssign for Matrix { fn sub_assign } matrix_view_assign_subtraction_iter "Elementwise assigning subtraction for two referenced matrices");

macro_rules! matrix_reference_matrix_value_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<Matrix<T>> for &Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                $implementation::<T, Matrix<T>, Matrix<T>>(self, &rhs)
            }
        }
    };
}

macro_rules! matrix_reference_matrix_value_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<Matrix<T>> for &Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter(),
                    self.size(),
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_reference_matrix_value_operation_iter!(impl Add for Matrix { fn add } matrix_view_addition_iter "Elementwise addition for two matrices with one referenced");
matrix_reference_matrix_value_operation_iter!(impl Sub for Matrix { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for two matrices with one referenced");
matrix_reference_matrix_value_operation!(impl Mul for Matrix { fn mul } matrix_view_multiplication "Matrix multiplication for two matrices with one referenced");

macro_rules! matrix_assign_matrix_value_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<Matrix<T>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            #[track_caller]
            #[inline]
            fn $method(&mut self, rhs: Matrix<T>) {
                let left_size = self.size();
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter_mut(),
                    left_size,
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_assign_matrix_value_operation_iter!(impl AddAssign for Matrix { fn add_assign } matrix_view_assign_addition_iter "Elementwise assigning addition for two matrices with one referenced");
matrix_assign_matrix_value_operation_iter!(impl SubAssign for Matrix { fn sub_assign } matrix_view_assign_subtraction_iter "Elementwise assigning subtraction for two matrices with one referenced");

macro_rules! matrix_value_matrix_reference_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<&Matrix<T>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Matrix<T>) -> Self::Output {
                $implementation::<T, Matrix<T>, Matrix<T>>(&self, rhs)
            }
        }
    };
}

macro_rules! matrix_value_matrix_reference_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<&Matrix<T>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Matrix<T>) -> Self::Output {
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter(),
                    self.size(),
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_value_matrix_reference_operation_iter!(impl Add for Matrix { fn add } matrix_view_addition_iter "Elementwise addition for two matrices with one referenced");
matrix_value_matrix_reference_operation_iter!(impl Sub for Matrix { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for two matrices with one referenced");
matrix_value_matrix_reference_operation!(impl Mul for Matrix { fn mul } matrix_view_multiplication "Matrix multiplication for two matrices with one referenced");

macro_rules! matrix_value_matrix_value_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<Matrix<T>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                $implementation::<T, Matrix<T>, Matrix<T>>(&self, &rhs)
            }
        }
    };
}

macro_rules! matrix_value_matrix_value_operation_iter {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl<T> $op<Matrix<T>> for Matrix<T>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                $implementation::<T, _, _>(
                    self.direct_row_major_reference_iter(),
                    self.size(),
                    rhs.direct_row_major_reference_iter(),
                    rhs.size(),
                )
            }
        }
    };
}

matrix_value_matrix_value_operation_iter!(impl Add for Matrix { fn add } matrix_view_addition_iter "Elementwise addition for two matrices");
matrix_value_matrix_value_operation_iter!(impl Sub for Matrix { fn sub } matrix_view_subtraction_iter "Elementwise subtraction for two matrices");
matrix_value_matrix_value_operation!(impl Mul for Matrix { fn mul } matrix_view_multiplication "Matrix multiplication for two matrices");

#[test]
fn test_all_16_combinations() {
    fn matrix() -> Matrix<i8> {
        Matrix::from_scalar(1)
    }
    fn matrix_view() -> MatrixView<i8, Matrix<i8>> {
        MatrixView::from(Matrix::from_scalar(1))
    }
    let mut results = Vec::with_capacity(16);
    results.push(matrix() + matrix());
    results.push(matrix() + &matrix());
    results.push(&matrix() + matrix());
    results.push(&matrix() + &matrix());
    results.push(matrix_view() + matrix());
    results.push(matrix_view() + &matrix());
    results.push(&matrix_view() + matrix());
    results.push(&matrix_view() + &matrix());
    results.push(matrix() + matrix_view());
    results.push(matrix() + &matrix_view());
    results.push(&matrix() + matrix_view());
    results.push(&matrix() + &matrix_view());
    results.push(matrix_view() + matrix_view());
    results.push(matrix_view() + &matrix_view());
    results.push(&matrix_view() + matrix_view());
    results.push(&matrix_view() + &matrix_view());
    for total in results {
        assert_eq!(total.scalar(), 2);
    }
}

#[test]
fn elementwise_addition_assign_test_all_8_combinations() {
    fn matrix() -> Matrix<i8> {
        Matrix::from_scalar(1)
    }
    fn matrix_view() -> MatrixView<i8, Matrix<i8>> {
        MatrixView::from(Matrix::from_scalar(1))
    }
    let mut results_matrix = Vec::with_capacity(16);
    let mut results_matrix_views = Vec::with_capacity(16);
    results_matrix.push({
        let mut x = matrix();
        x += matrix();
        x
    });
    results_matrix.push({
        let mut x = matrix();
        x += &matrix();
        x
    });
    results_matrix_views.push({
        let mut x = matrix_view();
        x += matrix();
        x
    });
    results_matrix_views.push({
        let mut x = matrix_view();
        x += &matrix();
        x
    });
    results_matrix.push({
        let mut x = matrix();
        x += matrix_view();
        x
    });
    results_matrix.push({
        let mut x = matrix();
        x += &matrix_view();
        x
    });
    results_matrix_views.push({
        let mut x = matrix_view();
        x += matrix_view();
        x
    });
    results_matrix_views.push({
        let mut x = matrix_view();
        x += &matrix_view();
        x
    });
    for total in results_matrix {
        assert_eq!(total.scalar(), 2);
    }
    for total in results_matrix_views {
        assert_eq!(total.get(0, 0), 2);
    }
}

#[test]
fn elementwise_subtraction_assign_test_all_8_combinations() {
    fn matrix() -> Matrix<i8> {
        Matrix::from_scalar(1)
    }
    fn matrix_view() -> MatrixView<i8, Matrix<i8>> {
        MatrixView::from(Matrix::from_scalar(1))
    }
    let mut results_matrix = Vec::with_capacity(16);
    let mut results_matrix_views = Vec::with_capacity(16);
    results_matrix.push({
        let mut x = matrix();
        x -= matrix();
        x
    });
    results_matrix.push({
        let mut x = matrix();
        x -= &matrix();
        x
    });
    results_matrix_views.push({
        let mut x = matrix_view();
        x -= matrix();
        x
    });
    results_matrix_views.push({
        let mut x = matrix_view();
        x -= &matrix();
        x
    });
    results_matrix.push({
        let mut x = matrix();
        x -= matrix_view();
        x
    });
    results_matrix.push({
        let mut x = matrix();
        x -= &matrix_view();
        x
    });
    results_matrix_views.push({
        let mut x = matrix_view();
        x -= matrix_view();
        x
    });
    results_matrix_views.push({
        let mut x = matrix_view();
        x -= &matrix_view();
        x
    });
    for total in results_matrix {
        assert_eq!(total.scalar(), 0);
    }
    for total in results_matrix_views {
        assert_eq!(total.get(0, 0), 0);
    }
}

/**
 * Elementwise negation for a referenced matrix.
 */
impl<T: Numeric> Neg for &Matrix<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    type Output = Matrix<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        self.map(|v| -v)
    }
}

/**
 * Elementwise negation for a matrix.
 */
impl<T: Numeric> Neg for Matrix<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    type Output = Matrix<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        -&self
    }
}

/**
 * Elementwise negation for a referenced matrix view.
 */
impl<T, S> Neg for &MatrixView<T, S>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S: MatrixRef<T>,
{
    type Output = Matrix<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        self.map(|v| -v)
    }
}

/**
 * Elementwise negation for a matrix view.
 */
impl<T, S> Neg for MatrixView<T, S>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S: MatrixRef<T>,
{
    type Output = Matrix<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        -&self
    }
}

macro_rules! matrix_scalar {
    (impl $op:tt for Matrix { fn $method:ident }) => {
        /**
         * Operation for a matrix and scalar by reference. The scalar is applied to
         * all elements, this is a shorthand for [map()](Matrix::map).
         */
        impl<T: Numeric> $op<&T> for &Matrix<T>
        where
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: &T) -> Self::Output {
                self.map(|x| (x).$method(rhs.clone()))
            }
        }

        /**
         * Operation for a matrix by value and scalar by reference. The scalar is applied to
         * all elements, this is a shorthand for [map()](Matrix::map).
         */
        impl<T: Numeric> $op<&T> for Matrix<T>
        where
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: &T) -> Self::Output {
                self.map(|x| (x).$method(rhs.clone()))
            }
        }

        /**
         * Operation for a matrix by reference and scalar by value. The scalar is applied to
         * all elements, this is a shorthand for [map()](Matrix::map).
         */
        impl<T: Numeric> $op<T> for &Matrix<T>
        where
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                self.map(|x| (x).$method(rhs.clone()))
            }
        }

        /**
         * Operation for a matrix and scalar by value. The scalar is applied to
         * all elements, this is a shorthand for [map()](Matrix::map).
         */
        impl<T: Numeric> $op<T> for Matrix<T>
        where
            for<'a> &'a T: NumericRef<T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                self.map(|x| (x).$method(rhs.clone()))
            }
        }
    };
}

macro_rules! matrix_view_scalar {
    (impl $op:tt for MatrixView { fn $method:ident }) => {
        /**
         * Operation for a matrix view and scalar by reference. The scalar is applied to
         * all elements, this is a shorthand for [map()](MatrixView::map).
         */
        impl<T, S> $op<&T> for &MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: &T) -> Self::Output {
                self.map(|x| (x).$method(rhs.clone()))
            }
        }

        /**
         * Operation for a matrix viiew by value and scalar by reference. The scalar is applied to
         * all elements, this is a shorthand for [map()](MatrixView::map).
         */
        impl<T, S> $op<&T> for MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: &T) -> Self::Output {
                self.map(|x| (x).$method(rhs.clone()))
            }
        }

        /**
         * Operation for a matrix view by reference and scalar by value. The scalar is applied to
         * all elements, this is a shorthand for [map()](MatrixView::map).
         */
        impl<T, S> $op<T> for &MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                self.map(|x| (x).$method(rhs.clone()))
            }
        }

        /**
         * Operation for a matrix view and scalar by value. The scalar is applied to
         * all elements, this is a shorthand for [map()](MatrixView::map).
         */
        impl<T, S> $op<T> for MatrixView<T, S>
        where
            T: Numeric,
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;
            #[inline]
            fn $method(self, rhs: T) -> Self::Output {
                self.map(|x| (x).$method(rhs.clone()))
            }
        }
    };
}

matrix_scalar!(impl Add for Matrix { fn add });
matrix_scalar!(impl Sub for Matrix { fn sub });
matrix_scalar!(impl Mul for Matrix { fn mul });
matrix_scalar!(impl Div for Matrix { fn div });

matrix_view_scalar!(impl Add for MatrixView { fn add });
matrix_view_scalar!(impl Sub for MatrixView { fn sub });
matrix_view_scalar!(impl Mul for MatrixView { fn mul });
matrix_view_scalar!(impl Div for MatrixView { fn div });
