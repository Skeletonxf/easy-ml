use crate::matrices::{Column, Row, Matrix};
use crate::matrices::views::{MatrixView, MatrixRef};
use crate::matrices::iterators::RowMajorReferenceIterator;
use crate::numeric::{Numeric, NumericRef};

use std::ops::{Add, Sub};

#[inline]
fn size<T, S>(matrix: &S) -> (Row, Column)
where
    S: MatrixRef<T>,
{
    (matrix.view_rows(), matrix.view_columns())
}

#[track_caller]
#[inline]
fn matrix_view_addition<T, S1, S2>(left: &S1, right: &S2) -> Matrix<T>
where
    for<'a> &'a T: NumericRef<T>,
    S1: MatrixRef<T>,
    S2: MatrixRef<T>,
{
    // LxM + LxM -> LxM
    assert!(size(left) == size(right),
        "Mismatched matrices, left is {}x{}, right is {}x{}, + is only defined for MxN + MxN",
        left.view_rows(), left.view_columns(), right.view_rows(), right.view_columns());

    let values = RowMajorReferenceIterator::from(left)
        .zip(RowMajorReferenceIterator::from(right))
        .map(|(x, y)| x + y)
        .collect();
    Matrix::from_flat_row_major(size(left), values)
}

#[track_caller]
#[inline]
fn matrix_view_subtraction<T, S1, S2>(left: &S1, right: &S2) -> Matrix<T>
where
    for<'a> &'a T: NumericRef<T>,
    S1: MatrixRef<T>,
    S2: MatrixRef<T>,
{
    // LxM + LxM -> LxM
    assert!(size(left) == size(right),
        "Mismatched matrices, left is {}x{}, right is {}x{}, - is only defined for MxN + MxN",
        left.view_rows(), left.view_columns(), right.view_rows(), right.view_columns());

    let values = RowMajorReferenceIterator::from(left)
        .zip(RowMajorReferenceIterator::from(right))
        .map(|(x, y)| x - y)
        .collect();
    Matrix::from_flat_row_major(size(left), values)
}

macro_rules! matrix_view_reference_reference_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S1, S2> $op<&MatrixView<T, S2>> for &MatrixView<T, S1>
        where
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T>,
            S2: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self.source_ref(), rhs.source_ref())
            }
        }
    }
}

matrix_view_reference_reference_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for two referenced matrix views");
matrix_view_reference_reference_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise subtraction for two referenced matrix views");

macro_rules! matrix_view_reference_value_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S1, S2> $op<MatrixView<T, S2>> for &MatrixView<T, S1>
        where
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T>,
            S2: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self.source_ref(), rhs.source_ref())
            }
        }
    }
}

matrix_view_reference_value_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for two matrix views with one referenced.");
matrix_view_reference_value_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise subtraction for two matrix views with one referenced.");

macro_rules! matrix_view_value_reference_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S1, S2> $op<&MatrixView<T, S2>> for MatrixView<T, S1>
        where
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T>,
            S2: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self.source_ref(), rhs.source_ref())
            }
        }
    }
}

matrix_view_value_reference_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for two matrix views with one referenced.");
matrix_view_value_reference_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise addition for two matrix views with one referenced.");

macro_rules! matrix_view_value_value_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S1, S2> $op<MatrixView<T, S2>> for MatrixView<T, S1>
        where
            for<'a> &'a T: NumericRef<T>,
            S1: MatrixRef<T>,
            S2: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S2>) -> Self::Output {
                $implementation::<T, S1, S2>(self.source_ref(), rhs.source_ref())
            }
        }
    }
}

matrix_view_value_value_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for two matrix views");
matrix_view_value_value_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise subtraction for two matrix views");
