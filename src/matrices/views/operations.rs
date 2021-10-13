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

// TODO: May want to add a default method to MatrixRef for returning a row major reference iterator
// that Matrix can override to provide a simpler implementation of.
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

macro_rules! matrix_view_reference_matrix_view_reference_operation {
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

matrix_view_reference_matrix_view_reference_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for two referenced matrix views");
matrix_view_reference_matrix_view_reference_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise subtraction for two referenced matrix views");

macro_rules! matrix_view_reference_matrix_view_value_operation {
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

matrix_view_reference_matrix_view_value_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for two matrix views with one referenced.");
matrix_view_reference_matrix_view_value_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise subtraction for two matrix views with one referenced.");

macro_rules! matrix_view_value_matrix_view_reference_operation {
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

matrix_view_value_matrix_view_reference_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for two matrix views with one referenced.");
matrix_view_value_matrix_view_reference_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise addition for two matrix views with one referenced.");

macro_rules! matrix_view_value_matrix_view_value_operation {
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

matrix_view_value_matrix_view_value_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for two matrix views");
matrix_view_value_matrix_view_value_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise subtraction for two matrix views");

macro_rules! matrix_view_reference_matrix_reference_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S> $op<&Matrix<T>> for &MatrixView<T, S>
        where
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Matrix<T>) -> Self::Output {
                $implementation::<T, S, Matrix<T>>(self.source_ref(), rhs)
            }
        }
    }
}

matrix_view_reference_matrix_reference_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for a referenced matrix view and a referenced matrix");
matrix_view_reference_matrix_reference_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise subtraction for a referenced matrix view and a referenced matrix");

macro_rules! matrix_view_reference_matrix_value_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S> $op<Matrix<T>> for &MatrixView<T, S>
        where
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                $implementation::<T, S, Matrix<T>>(self.source_ref(), &rhs)
            }
        }
    }
}

matrix_view_reference_matrix_value_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for a referenced matrix view and a matrix");
matrix_view_reference_matrix_value_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise subtraction for a referenced matrix view and a matrix");

macro_rules! matrix_view_value_matrix_reference_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S> $op<&Matrix<T>> for MatrixView<T, S>
        where
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &Matrix<T>) -> Self::Output {
                $implementation::<T, S, Matrix<T>>(self.source_ref(), rhs)
            }
        }
    }
}

matrix_view_value_matrix_reference_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for a matrix view and a referenced matrix");
matrix_view_value_matrix_reference_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise subtraction for a matrix view and a referenced matrix");

macro_rules! matrix_view_value_matrix_value_operation {
    (impl $op:tt for MatrixView { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S> $op<Matrix<T>> for MatrixView<T, S>
        where
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: Matrix<T>) -> Self::Output {
                $implementation::<T, S, Matrix<T>>(self.source_ref(), &rhs)
            }
        }
    }
}

matrix_view_value_matrix_value_operation!(impl Add for MatrixView { fn add } matrix_view_addition "Elementwise addition for a matrix view and a matrix");
matrix_view_value_matrix_value_operation!(impl Sub for MatrixView { fn sub } matrix_view_subtraction "Elementwise subtraction for a matrix view and a matrix");

macro_rules! matrix_reference_matrix_view_reference_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S> $op<&MatrixView<T, S>> for &Matrix<T>
        where
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S>) -> Self::Output {
                $implementation::<T, Matrix<T>, S>(self, rhs.source_ref())
            }
        }
    }
}

matrix_reference_matrix_view_reference_operation!(impl Add for Matrix { fn add } matrix_view_addition "Elementwise addition for a referenced matrix and a referenced matrix view");
matrix_reference_matrix_view_reference_operation!(impl Sub for Matrix { fn sub } matrix_view_subtraction "Elementwise subtraction for a referenced matrix and a referenced matrix view");

macro_rules! matrix_reference_matrix_view_value_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S> $op<MatrixView<T, S>> for &Matrix<T>
        where
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S>) -> Self::Output {
                $implementation::<T, Matrix<T>, S>(self, rhs.source_ref())
            }
        }
    }
}

matrix_reference_matrix_view_value_operation!(impl Add for Matrix { fn add } matrix_view_addition "Elementwise addition for a referenced matrix and a matrix view");
matrix_reference_matrix_view_value_operation!(impl Sub for Matrix { fn sub } matrix_view_subtraction "Elementwise subtraction for a referenced matrix and a matrix view");

macro_rules! matrix_value_matrix_view_reference_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S> $op<&MatrixView<T, S>> for Matrix<T>
        where
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: &MatrixView<T, S>) -> Self::Output {
                $implementation::<T, Matrix<T>, S>(&self, rhs.source_ref())
            }
        }
    }
}

matrix_value_matrix_view_reference_operation!(impl Add for Matrix { fn add } matrix_view_addition "Elementwise addition for a matrix and a referenced matrix view");
matrix_value_matrix_view_reference_operation!(impl Sub for Matrix { fn sub } matrix_view_subtraction "Elementwise subtraction for a matrix and a referenced matrix view");

macro_rules! matrix_value_matrix_view_value_operation {
    (impl $op:tt for Matrix { fn $method:ident } $implementation:ident $doc:tt) => {
        #[doc=$doc]
        impl <T: Numeric, S> $op<MatrixView<T, S>> for Matrix<T>
        where
            for<'a> &'a T: NumericRef<T>,
            S: MatrixRef<T>,
        {
            type Output = Matrix<T>;

            #[track_caller]
            #[inline]
            fn $method(self, rhs: MatrixView<T, S>) -> Self::Output {
                $implementation::<T, Matrix<T>, S>(&self, rhs.source_ref())
            }
        }
    }
}

matrix_value_matrix_view_value_operation!(impl Add for Matrix { fn add } matrix_view_addition "Elementwise addition for a matrix and a matrix view");
matrix_value_matrix_view_value_operation!(impl Sub for Matrix { fn sub } matrix_view_subtraction "Elementwise subtraction for a matrix and a matrix view");

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
