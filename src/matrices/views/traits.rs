/*!
 * Trait implementations for [MatrixRef] and [MatrixMut].
 *
 * These implementations are written here but Rust docs will display them on the
 * traits' pages.
 *
 * An owned or referenced [Matrix] is a MatrixRef, and a MatrixMut if not a shared
 * reference, Therefore, you can pass a Matrix to any function which takes a MatrixRef.
 *
 * Boxed MatrixRef and MatrixMut values also implement MatrixRef and MatrixMut respectively.
 *
 * All MatrixRef and MatrixMut implementations for Matrices are also [NoInteriorMutability].
 *
 * Since a Matrix always stores its data in row major order,
 * [`data_layout()`](MatrixRef::data_layout) will return
 * [`DataLayout::RowMajor`], but third party matrix types implementing
 * MatrixRef/MatrixMut may use a column major layout.
 */

use crate::matrices::views::{DataLayout, MatrixMut, MatrixRef, NoInteriorMutability};
use crate::matrices::{Column, Matrix, Row};

// # Safety
//
// The type implementing MatrixRef must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement MatrixRef
// correctly as well.
/**
 * If some type implements MatrixRef, then a reference to it implements MatrixRef as well
 */
unsafe impl<T, S> MatrixRef<T> for &S
where
    S: MatrixRef<T>,
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        MatrixRef::try_get_reference(*self, row, column)
    }

    fn view_rows(&self) -> Row {
        MatrixRef::view_rows(*self)
    }

    fn view_columns(&self) -> Column {
        MatrixRef::view_columns(*self)
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        unsafe { MatrixRef::get_reference_unchecked(*self, row, column) }
    }

    fn data_layout(&self) -> DataLayout {
        DataLayout::RowMajor
    }
}

// # Safety
//
// The type implementing NoInteriorMutability must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement
// NoInteriorMutability correctly as well.
/**
 * If some type implements NoInteriorMutability, then a reference to it implements
 * NoInteriorMutability as well. The reverse would not be true, since a type that does
 * have interior mutability would remain interiorly mutable behind a shared reference. However,
 * since this type promises not to have interior mutability, taking a shared reference can't
 * introduce any.
 */
unsafe impl<S> NoInteriorMutability for &S where S: NoInteriorMutability {}

// # Safety
//
// The type implementing MatrixRef must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement MatrixRef
// correctly as well.
/**
 * If some type implements MatrixRef, then an exclusive reference to it implements MatrixRef
 * as well
 */
unsafe impl<T, S> MatrixRef<T> for &mut S
where
    S: MatrixRef<T>,
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        MatrixRef::try_get_reference(*self, row, column)
    }

    fn view_rows(&self) -> Row {
        MatrixRef::view_rows(*self)
    }

    fn view_columns(&self) -> Column {
        MatrixRef::view_columns(*self)
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        unsafe { MatrixRef::get_reference_unchecked(*self, row, column) }
    }

    fn data_layout(&self) -> DataLayout {
        DataLayout::RowMajor
    }
}

// # Safety
//
// The type implementing MatrixMut must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement MatrixMut
// correctly as well.
/**
 * If some type implements MatrixMut, then an exclusive reference to it implements MatrixMut
 * as well
 */
unsafe impl<T, S> MatrixMut<T> for &mut S
where
    S: MatrixMut<T>,
{
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        MatrixMut::try_get_reference_mut(*self, row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        unsafe { MatrixMut::get_reference_unchecked_mut(*self, row, column) }
    }
}

// # Safety
//
// The type implementing NoInteriorMutability must implement it correctly, so by delegating to it
// without changing any indexes or introducing interior mutability, we implement
// NoInteriorMutability correctly as well.
/**
 * If some type implements NoInteriorMutability, then an exclusive reference to it implements
 * NoInteriorMutability as well. The reverse would not be true, since a type that does
 * have interior mutability would remain interiorly mutable behind an exclusive reference. However,
 * since this type promises not to have interior mutability, taking an exclusive reference can't
 * introduce any.
 */
unsafe impl<S> NoInteriorMutability for &mut S {}

// # Safety
//
// Since we hold an owned Matrix we know it is not possible to mutate the size of the matrix
// out from under us.
/**
 * A Matrix implements MatrixRef.
 */
unsafe impl<T> MatrixRef<T> for Matrix<T> {
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
        unsafe { Matrix::_get_reference_unchecked(self, row, column) }
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
 * A Matrix implements MatrixMut.
 */
unsafe impl<T> MatrixMut<T> for Matrix<T> {
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        Matrix::_try_get_reference_mut(self, row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        unsafe { Matrix::_get_reference_unchecked_mut(self, row, column) }
    }
}

// # Safety
//
// We promise to never implement interior mutability for Matrix.
/**
 * A Matrix implements NoInteriorMutability.
 */
unsafe impl<T> NoInteriorMutability for Matrix<T> {}

// # Safety
//
// Since the MatrixRef we box must implement MatrixRef correctly, so do we by delegating to it,
// as a box doesn't introduce any interior mutability.
/**
 * A box of a MatrixRef also implements MatrixRef.
 */
unsafe impl<T, S> MatrixRef<T> for Box<S>
where
    S: MatrixRef<T>,
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
        unsafe { self.as_ref().get_reference_unchecked(row, column) }
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
unsafe impl<T, S> MatrixMut<T> for Box<S>
where
    S: MatrixMut<T>,
{
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        self.as_mut().try_get_reference_mut(row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        unsafe { self.as_mut().get_reference_unchecked_mut(row, column) }
    }
}

// # Safety
//
// Box doesn't introduce any interior mutability, so we can implement if it the type we box does.
/**
 * A box of a NoInteriorMutability also implements NoInteriorMutability
 */
unsafe impl<S> NoInteriorMutability for Box<S> where S: NoInteriorMutability {}

// # Safety
//
// Since the MatrixRef we box must implement MatrixRef correctly, so do we by delegating to it,
// as a box doesn't introduce any interior mutability.
/**
 * A box of a dynamic MatrixRef also implements MatrixRef.
 */
unsafe impl<T> MatrixRef<T> for Box<dyn MatrixRef<T>> {
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
        unsafe { self.as_ref().get_reference_unchecked(row, column) }
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
unsafe impl<T> MatrixRef<T> for Box<dyn MatrixMut<T>> {
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
        unsafe { self.as_ref().get_reference_unchecked(row, column) }
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
unsafe impl<T> MatrixMut<T> for Box<dyn MatrixMut<T>> {
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        self.as_mut().try_get_reference_mut(row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        unsafe { self.as_mut().get_reference_unchecked_mut(row, column) }
    }
}

// # Safety
//
// Box doesn't introduce any interior mutability, so we can implement if it the type we box does.
/**
 * A box of a dynamic NoInteriorMutability also implements NoInteriorMutability
 */
unsafe impl NoInteriorMutability for Box<dyn NoInteriorMutability> {}

// # Safety
//
// Box doesn't introduce any interior mutability, so we can implement if it the type we box does.
/**
 * A box of a dynamic MatrixRef also implements NoInteriorMutability, since NoInteriorMutability
 * is supertrait of MatrixRef
 */
unsafe impl<T> NoInteriorMutability for Box<dyn MatrixRef<T>> {}

// # Safety
//
// Box doesn't introduce any interior mutability, so we can implement if it the type we box does.
/**
 * A box of a dynamic MatrixMut also implements NoInteriorMutability, since NoInteriorMutability
 * is supertrait of MatrixMut
 */
unsafe impl<T> NoInteriorMutability for Box<dyn MatrixMut<T>> {}
