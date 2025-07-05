use crate::matrices::views::{DataLayout, MatrixMut, MatrixRef, NoInteriorMutability};
use crate::matrices::{Column, Row};
use crate::tensors::views::reverse_indexes;

use std::marker::PhantomData;

/**
 * A view over a matrix where some or all of the rows and columns are iterated in reverse order.
 *
 * ```
 * use easy_ml::matrices::Matrix;
 * use easy_ml::matrices::views::{MatrixView, MatrixReverse, Reverse};
 * let ab = Matrix::from(vec![
 *     vec![ 0, 1, 2 ],
 *     vec![ 3, 4, 5 ]
 * ]);
 * let reversed = ab.reverse(Reverse { rows: true, ..Default::default() });
 * let also_reversed = MatrixView::from(
 *     MatrixReverse::from(&ab, Reverse { rows: true, columns: false })
 * );
 * assert_eq!(reversed, also_reversed);
 * assert_eq!(
 *     reversed,
 *     Matrix::from(vec![
 *         vec![ 3, 4, 5 ],
 *         vec![ 0, 1, 2 ]
 *     ])
 * );
 * ```
 */
#[derive(Clone, Debug)]
pub struct MatrixReverse<T, S> {
    source: S,
    rows: bool,
    columns: bool,
    _type: PhantomData<T>,
}

/**
 * Helper struct for declaring which of `rows` and `columns` should be reversed for iteration.
 *
 * If a dimension is set to `false` it will iterate in its normal order. If a dimension is
 * set to `true` the iteration order will be reversed, so the first index 0 becomes the last
 * length-1, and the last index length-1 becomes 0
 */
// NB: Default impl for bool is false, which is what we want here
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct Reverse {
    pub rows: bool,
    pub columns: bool,
}

impl<T, S> MatrixReverse<T, S>
where
    S: MatrixRef<T>,
{
    /**
     * Creates a MatrixReverse from a source and a struct for which dimensions to reverse the
     * order of iteration for. If either or both of rows and columns in [Reverse] are set to false
     * the iteration order for that dimension will continue to iterate in its normal order.
     */
    pub fn from(source: S, reverse: Reverse) -> MatrixReverse<T, S> {
        MatrixReverse {
            source,
            rows: reverse.rows,
            columns: reverse.columns,
            _type: PhantomData,
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
unsafe impl<T, S> NoInteriorMutability for MatrixReverse<T, S> where S: NoInteriorMutability {}

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
        // If the Matrix has 0 length rows or columns the Tensor reverse_indexes function
        // would reach out of bounds as it does not need to handle this case for tensors.
        // Since the caller can expect to be able to query a 0x0 matrix and get None for
        // any index, we must ensure this out of bounds calculation doesn't happen.
        if self.source.view_rows() == 0 || self.source.view_columns() == 0 {
            return None;
        }
        let [row, column] = reverse_indexes(
            &[row, column],
            &[
                ("row", self.source.view_rows()),
                ("column", self.source.view_columns()),
            ],
            &[self.rows, self.columns],
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
        unsafe {
            // It is the caller's responsibiltiy to call this unsafe function with only valid
            // indexes. If the source matrix is not at least 1x1, there are no valid indexes and hence
            // the caller must not call this function.
            // Given we can assume the matrix is at least 1x1 if we're being called, this calculation
            // will return a new index which is also in range if the input was, so we won't
            // introduce any out of bounds reads.
            let [row, column] = reverse_indexes(
                &[row, column],
                &[
                    ("row", self.source.view_rows()),
                    ("column", self.source.view_columns()),
                ],
                &[self.rows, self.columns],
            );
            self.source.get_reference_unchecked(row, column)
        }
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
        // If the Matrix has 0 length rows or columns the Tensor reverse_indexes function
        // would reach out of bounds as it does not need to handle this case for tensors.
        // Since the caller can expect to be able to query a 0x0 matrix and get None for
        // any index, we must ensure this out of bounds calculation doesn't happen.
        if self.source.view_rows() == 0 || self.source.view_columns() == 0 {
            return None;
        }
        let [row, column] = reverse_indexes(
            &[row, column],
            &[
                ("row", self.source.view_rows()),
                ("column", self.source.view_columns()),
            ],
            &[self.rows, self.columns],
        );
        self.source.try_get_reference_mut(row, column)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        unsafe {
            // It is the caller's responsibiltiy to call this unsafe function with only valid
            // indexes. If the source matrix is not at least 1x1, there are no valid indexes and hence
            // the caller must not call this function.
            // Given we can assume the matrix is at least 1x1 if we're being called, this calculation
            // will return a new index which is also in range if the input was, so we won't
            // introduce any out of bounds reads.
            let [row, column] = reverse_indexes(
                &[row, column],
                &[
                    ("row", self.source.view_rows()),
                    ("column", self.source.view_columns()),
                ],
                &[self.rows, self.columns],
            );
            self.source.get_reference_unchecked_mut(row, column)
        }
    }
}
