use crate::matrices::views::{DataLayout, MatrixMut, MatrixRef, NoInteriorMutability};
use crate::matrices::{Column, Row};
use crate::tensors::views::reverse_indexes;

use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct MatrixReverse<T, S> {
    source: S,
    rows: bool,
    columns: bool,
    _type: PhantomData<T>,
}

/**
 * Helper struct for declaring which of `rows` and `columns` should be reversed for iteration.
 * If a dimension is set to `false` it will iterate in its normal order. If a dimension is
 * set to `true` the iteration order will be reversed, so the first index 0 becomes the last
 * length-1, and the last index length-1 becomes 0
 */
// NB: Default impl for bool is false, which is what we want here
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct Reverse {
    rows: bool,
    columns: bool,
}

impl<T, S> MatrixReverse<T, S>
where
    S: MatrixRef<T>,
{
    /**
     * Creates a MatrixReverse from a source and a struct for which dimensions to reverse the
     * order of iteration for. If either or both of rows and columns in [Reverse] are set to false
     * the iteration order for that dimension will be will continue to iterate in its normal
     * order.
     */
    pub fn from(source: S, reverse: Reverse) -> MatrixReverse<T, S> {
        MatrixReverse {
            source,
            rows: reverse.rows,
            columns: reverse.columns,
            _type: PhantomData
        }
    }

    /**
     * Consumes the MatrixReverse, yielding the source it was created from.
     */
    #[allow(dead_code)]
    pub fn source(self) -> S {
        self.source
    }

    /**
     * Gives a reference to the MatrixReverse's source (in which the data is not reversed).
     */
    #[allow(dead_code)]
    pub fn source_ref(&self) -> &S {
        &self.source
    }

    /**
     * Gives a mutable reference to the MatrixReverse's source (in which the data is not reversed).
     */
    #[allow(dead_code)]
    pub fn source_ref_mut(&mut self) -> &mut S {
        &mut self.source
    }
}

// # Safety
//
// Since our source implements NoInteriorMutability correctly, so do we by delegating to it, as
// we don't introduce any interior mutability.
/**
 * A MatrixReverse of a NoInteriorMutability type implements NoInteriorMutability.
 */
unsafe impl<T, S> NoInteriorMutability for MatrixReverse<T, S>
where
    S: NoInteriorMutability,
{}

// # Safety
//
// The type implementing MatrixRef must implement it correctly, so by delegating to it
// by only reversing some indexes and not introducing interior mutability, we implement
// MatrixRef correctly as well.
/**
 * A MatrixReverse implements MatrixRef, with the dimension names the MatrixReverse was created
 * with iterating in reverse order compared to the dimension names in the original source.
 */
unsafe impl<T, S> MatrixRef<T> for MatrixReverse<T, S>
where
    S: MatrixRef<T>,
{
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        let [row, column] = reverse_indexes(
            &[row, column],
            &[("row", row), ("column", column)],
            &[self.rows, self.columns]
        );
        self.source.try_get_reference(row, column)
    }

    fn view_rows(&self) -> Row {
        self.source.view_rows()
    }

    fn view_columns(&self) -> Column {
        self.source.view_columns()
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        // Matrices must always be at least 1x1, this was not as explicitly stated on the
        // MatrixRef trait prior to 2.0 but clearly mentioned on the Matrix struct and implied by
        // the documentation about indexing. Given we can assume the matrix is at least 1x1, this
        // calculation will return a new index which is also in range if the input was, so we won't
        // introduce any out of bounds reads.
        let [row, column] = reverse_indexes(
            &[row, column],
            &[("row", row), ("column", column)],
            &[self.rows, self.columns]
        );
        self.source.get_reference_unchecked(row, column)
    }

    fn data_layout(&self) -> DataLayout {
        // There might be some specific cases where reversing maintains a linear order but
        // in general I think reversing only some indexes is going to mean any attempt at being
        // able to take a slice that matches up with our view_shape is gone.
        DataLayout::Other
    }
}

// # Safety
//
// The type implementing MatrixMut must implement it correctly, so by delegating to it
// by only reversing some indexes and not introducing interior mutability, we implement
// MatrixMut correctly as well.
/**
 * A MatrixReverse implements MatrixMut, with the dimension names the MatrixReverse was created
 * with iterating in reverse order compared to the dimension names in the original source.
 */
unsafe impl<T, S> MatrixMut<T> for MatrixReverse<T, S>
where
    S: MatrixMut<T>,
{
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        let [row, column] = reverse_indexes(
            &[row, column],
            &[("row", row), ("column", column)],
            &[self.rows, self.columns]
        );
        self.source.try_get_reference_mut(row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        // Matrices must always be at least 1x1, this was not as explicitly stated on the
        // MatrixRef trait prior to 2.0 but clearly mentioned on the Matrix struct and implied by
        // the documentation about indexing. Given we can assume the matrix is at least 1x1, this
        // calculation will return a new index which is also in range if the input was, so we won't
        // introduce any out of bounds reads.
        let [row, column] = reverse_indexes(
            &[row, column],
            &[("row", row), ("column", column)],
            &[self.rows, self.columns]
        );
        self.source.get_reference_unchecked_mut(row, column)
    }
}
