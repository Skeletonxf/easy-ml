/*!
 * Trait implementations for [MatrixRef](MatrixRef) and [MatrixMut](MatrixMut).
 *
 * These implementations are written here but Rust docs will display them on the
 * traits' pages.
 *
 * An owned or referenced [Matrix](Matrix) is a MatrixRef, and a MatrixMut if not a shared
 * reference, Therefore, you can pass a Matrix to any function which takes a MatrixRef.
 *
 * Boxed MatrixRef and MatrixMut values also implement MatrixRef and MatrixMut respectively.
 *
 * Since a Matrix always stores its data in row major order,
 * [`data_layout()`](MatrixRef::data_layout) will return
 * [`DataLayout::RowMajor`](DataLayout::RowMajor), but third party matrix types implementing
 * MatrixRef/MatrixMut may use a column major layout.
 */

use crate::matrices::{Column, Matrix, Row};
use crate::matrices::views::{MatrixRef, MatrixMut, DataLayout};

// # Safety
//
// Since we hold a shared reference to a Matrix and Matrix does not implement interior mutability
// we know it is not possible to mutate the size of the matrix out from under us.
/**
 * A shared reference to a Matrix implements MatrixRef.
 */
unsafe impl <'source, T> MatrixRef<T> for &'source Matrix<T> {
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        Matrix::_try_get_reference(self, row, column)
    }

    fn view_rows(&self) -> Row {
        Matrix::rows(self)
    }

    fn view_columns(&self) -> Column {
        Matrix::columns(self)
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        Matrix::_get_reference_unchecked(self, row, column)
    }

    fn data_layout(&self) -> DataLayout {
        DataLayout::RowMajor
    }
}

// # Safety
//
// Since we hold an exclusive reference to a Matrix we know it is not possible to mutate
// the size of the matrix out from under us.
/**
 * An exclusive reference to a Matrix implements MatrixRef.
 */
unsafe impl <'source, T> MatrixRef<T> for &'source mut Matrix<T> {
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        Matrix::_try_get_reference(self, row, column)
    }

    fn view_rows(&self) -> Row {
        Matrix::rows(self)
    }

    fn view_columns(&self) -> Column {
        Matrix::columns(self)
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        Matrix::_get_reference_unchecked(self, row, column)
    }

    fn data_layout(&self) -> DataLayout {
        DataLayout::RowMajor
    }
}

// # Safety
//
// Since we hold an exclusive reference to a Matrix we know it is not possible to mutate
// the size of the matrix out from under us.
/**
 * An exclusive reference to a Matrix implements MatrixMut.
 */
unsafe impl <'source, T> MatrixMut<T> for &'source mut Matrix<T> {
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        Matrix::_try_get_reference_mut(self, row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        Matrix::_get_reference_unchecked_mut(self, row, column)
    }
}

// # Safety
//
// Since we hold an owned Matrix we know it is not possible to mutate the size of the matrix
// out from under us.
/**
 * An owned Matrix implements MatrixRef.
 */
unsafe impl <T> MatrixRef<T> for Matrix<T> {
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        Matrix::_try_get_reference(self, row, column)
    }

    fn view_rows(&self) -> Row {
        Matrix::rows(self)
    }

    fn view_columns(&self) -> Column {
        Matrix::columns(self)
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        Matrix::_get_reference_unchecked(self, row, column)
    }

    fn data_layout(&self) -> DataLayout {
        DataLayout::RowMajor
    }
}

// # Safety
//
// Since we hold an owned Matrix we know it is not possible to mutate the size of the matrix
// out from under us.
/**
 * An owned Matrix implements MatrixMut.
 */
unsafe impl <T> MatrixMut<T> for Matrix<T> {
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        Matrix::_try_get_reference_mut(self, row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        Matrix::_get_reference_unchecked_mut(self, row, column)
    }
}

// # Safety
//
// Since the MatrixRef we box must implement MatrixRef correctly, so do we by delegating to it,
// as a box doesn't introduce any interior mutability.
/**
 * A box of a MatrixRef also implements MatrixRef.
 */
unsafe impl <T, S> MatrixRef<T> for Box<S>
where
    S: MatrixRef<T>
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        self.as_ref().try_get_reference(row, column)
    }

    fn view_rows(&self) -> Row {
        self.as_ref().view_rows()
    }

    fn view_columns(&self) -> Column {
        self.as_ref().view_columns()
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        self.as_ref().get_reference_unchecked(row, column)
    }

    fn data_layout(&self) -> DataLayout {
        self.as_ref().data_layout()
    }
}

// # Safety
//
// Since the MatrixMut we box must implement MatrixMut correctly, so do we by delegating to it,
// as a box doesn't introduce any interior mutability.
/**
 * A box of a MatrixMut also implements MatrixMut.
 */
unsafe impl <T, S> MatrixMut<T> for Box<S>
where
    S: MatrixMut<T>
{
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        self.as_mut().try_get_reference_mut(row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        self.as_mut().get_reference_unchecked_mut(row, column)
    }
}

// # Safety
//
// Since the MatrixRef we box must implement MatrixRef correctly, so do we by delegating to it,
// as a box doesn't introduce any interior mutability.
/**
 * A box of a dynamic MatrixRef also implements MatrixRef.
 */
unsafe impl <T> MatrixRef<T> for Box<dyn MatrixRef<T>> {
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        self.as_ref().try_get_reference(row, column)
    }

    fn view_rows(&self) -> Row {
        self.as_ref().view_rows()
    }

    fn view_columns(&self) -> Column {
        self.as_ref().view_columns()
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        self.as_ref().get_reference_unchecked(row, column)
    }

    fn data_layout(&self) -> DataLayout {
        self.as_ref().data_layout()
    }
}

// # Safety
//
// Since the MatrixMut we box must implement MatrixRef correctly, so do we by delegating to it,
// as a box doesn't introduce any interior mutability.
/**
 * A box of a dynamic MatrixMut also implements MatrixRef.
 */
unsafe impl <T> MatrixRef<T> for Box<dyn MatrixMut<T>> {
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        self.as_ref().try_get_reference(row, column)
    }

    fn view_rows(&self) -> Row {
        self.as_ref().view_rows()
    }

    fn view_columns(&self) -> Column {
        self.as_ref().view_columns()
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        self.as_ref().get_reference_unchecked(row, column)
    }

    fn data_layout(&self) -> DataLayout {
        self.as_ref().data_layout()
    }
}

// # Safety
//
// Since the MatrixMut we box must implement MatrixMut correctly, so do we by delegating to it,
// as a box doesn't introduce any interior mutability.
/**
 * A box of a dynamic MatrixMut also implements MatrixMut.
 */
unsafe impl <T> MatrixMut<T> for Box<dyn MatrixMut<T>> {
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        self.as_mut().try_get_reference_mut(row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        self.as_mut().get_reference_unchecked_mut(row, column)
    }
}
