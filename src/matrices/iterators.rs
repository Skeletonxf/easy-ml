/*!
 * Iterators over parts of a Matrix
 *
 * # Examples
 *
 * Extending a matrix with new columns
 * ```
 * use easy_ml::matrices::Matrix;
 *
 * // we start with some matrix where the first and second columns correspond
 * // to x and y points
 * let mut matrix = Matrix::from(vec![
 *     vec![ 3.0, 4.0 ],
 *     vec![ 8.0, 1.0 ],
 *     vec![ 2.0, 9.0 ]]);
 * // insert a third column based on the formula x * y
 * matrix.insert_column_with(2, matrix.column_iter(0)
 *     // join together the x and y columns
 *     .zip(matrix.column_iter(1))
 *     // compute the values for the new column
 *     .map(|(x, y)| x * y)
 *     // Collect into a vector so we stop immutably borrowing from `matrix`.
 *     // This is only neccessary when we use the data from a Matrix to modify itself,
 *     // because the rust compiler enforces that we do not mutably and immutably borrow
 *     // something at the same time. If we used data from a different Matrix to update
 *     // `matrix` then we could stop at map and pass the iterator directly.
 *     .collect::<Vec<f64>>()
 *     // now that the Vec created owns the data for the new column and we have stopped
 *     // borrowing immutably from `matrix` we turn the vec back into an iterator and
 *     // mutably borrow `matrix` to add the new column
 *     .drain(..));
 * assert_eq!(matrix.get(0, 2), 3.0 * 4.0);
 * assert_eq!(matrix.get(1, 2), 8.0 * 1.0);
 * assert_eq!(matrix.get(2, 2), 2.0 * 9.0);
 * ```
 *
 * # Matrix layout and iterator performance
 *
 * Internally the Matrix type uses a flattened array with [row major storage](https://en.wikipedia.org/wiki/Row-_and_column-major_order)
 * of the data. Due to [CPU cache lines](https://stackoverflow.com/questions/3928995/how-do-cache-lines-work)
 * this means row major access of elements is likely to be faster than column major indexing,
 * once a matrix is large enough, so you should favor iterating through each row.
 *
 * ```
 * use easy_ml::matrices::Matrix;
 * let matrix = Matrix::from(vec![
 *    vec![ 1, 2 ],
 *    vec![ 3, 4 ]]);
 * // storage of elements is [1, 2, 3, 4]
 *
 * // row major access
 * for row in 0..2 {
 *     for column in 0..2 {
 *         println!("{}", matrix.get(row, column));
 *     }
 * } // -> 1 2 3 4
 * matrix.row_major_iter().for_each(|e| println!("{}", e)); // -> 1 2 3 4
 *
 * // column major access
 * for column in 0..2 {
 *     for row in 0..2 {
 *         println!("{}", matrix.get(row, column));
 *     }
 * } // -> 1 3 2 4
 * matrix.column_major_iter().for_each(|e| println!("{}", e)); // -> 1 3 2 4
 * ```
 */

use std::iter::{ExactSizeIterator, FusedIterator};
use std::marker::PhantomData;
use std::ops::Range;

use crate::matrices::{Matrix, Row, Column};
use crate::matrices::views::MatrixRef;

trait MatrixRefExtension<T>: MatrixRef<T> {
    /**
     * Helper for asserting starting indexes are in range.
     */
    fn index_is_valid(&self, row: Row, column: Column) -> bool {
        row < self.view_rows() && column < self.view_columns()
    }
}

impl <R, T> MatrixRefExtension<T> for R
where R: MatrixRef<T> {}

/**
 * An iterator over a column in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * Depending on the column iterator you want to obtain,
 * can either iterate through 1, 3 or 2, 4.
 */
#[derive(Debug)]
pub struct ColumnIterator<'a, T: Clone, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column: Column,
    range: Range<Row>,
    _type: PhantomData<&'a T>,
}

impl <'a, T: Clone> ColumnIterator<'a, T> {
    /**
     * Constructs a column iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>, column: Column) -> ColumnIterator<T> {
        assert!(matrix.index_is_valid(0, column), "Expected ({},{}) to be in range", 0, column);
        ColumnIterator {
            matrix,
            column,
            range: 0..matrix.view_rows(),
            _type: PhantomData,
        }
    }
}

impl <'a, T: Clone, S: MatrixRef<T>> Iterator for ColumnIterator<'a, T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(row) => unsafe {
                // Safety: We initialised the range to 0..matrix.view_rows(), and
                // checked we can read from this column at creation, hence if
                // the view_rows have not changed this read is in bounds and if
                // they are able to be changed, then the MatrixRef implementation is
                // required to bounds check for us.
                Some(self.matrix.get_reference_unchecked(row, self.column).clone())
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl <'a, T: Clone, S: MatrixRef<T>> FusedIterator for ColumnIterator<'a, T, S> {}
impl <'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for ColumnIterator<'a, T, S> {}

/**
 * An iterator over a row in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * Depending on the row iterator you want to obtain,
 * can either iterate through 1, 2 or 3, 4.
 */
#[derive(Debug)]
pub struct RowIterator<'a, T: Clone, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    row: Row,
    range: Range<Column>,
    _type: PhantomData<&'a T>,
}

impl <'a, T: Clone> RowIterator<'a, T> {
    /**
     * Constructs a row iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>, row: Row) -> RowIterator<T> {
        assert!(matrix.index_is_valid(row, 0), "Expected ({},{}) to be in range", row, 0);
        RowIterator {
            matrix,
            row,
            range: 0..matrix.view_columns(),
            _type: PhantomData,
        }
    }
}

impl <'a, T: Clone, S: MatrixRef<T>> Iterator for RowIterator<'a, T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(column) => unsafe {
                // Safety: We initialised the range to 0..matrix.view_columns(), and
                // checked we can read from this row at creation, hence if
                // the view_columns have not changed this read is in bounds and if
                // they are able to be changed, then the MatrixRef implementation is
                // required to bounds check for us.
                Some(self.matrix.get_reference_unchecked(self.row, column).clone())
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl <'a, T: Clone, S: MatrixRef<T>> FusedIterator for RowIterator<'a, T, S> {}
impl <'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for RowIterator<'a, T, S> {}

/**
 * A column major iterator over all values in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as 1, 3, 2, 4
 */
#[derive(Debug)]
pub struct ColumnMajorIterator<'a, T: Clone, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column_counter: Column,
    row_counter: Row,
    finished: bool,
    _type: PhantomData<&'a T>,
}

impl <'a, T: Clone> ColumnMajorIterator<'a, T> {
    /**
     * Constructs a column major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> ColumnMajorIterator<T> {
        assert!(matrix.index_is_valid(0, 0), "Expected ({},{}) to be in range", 0, 0);
        ColumnMajorIterator {
            matrix,
            column_counter: 0,
            row_counter: 0,
            finished: false,
            _type: PhantomData,
        }
    }
}

impl <'a, T: Clone, S: MatrixRef<T>> Iterator for ColumnMajorIterator<'a, T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = unsafe {
            // Safety: We checked on creation that 0,0 is in range, and after getting
            // our next value we check if we hit the end of the matrix and will avoid
            // calling this on our next loop if we finished. Hence if the view_size has
            // not changed this read is in bounds and if they are able to be changed,
            // then the MatrixRef implementation is required to bounds check for us.
            Some(self.matrix.get_reference_unchecked(self.row_counter, self.column_counter).clone())
        };

        if self.row_counter == self.matrix.view_rows() - 1
                && self.column_counter == self.matrix.view_columns() -1 {
            // reached end of matrix for next iteration
            self.finished = true;
        }

        if self.row_counter == self.matrix.view_rows() - 1 {
            // reached end of a column, need to reset to first element in next column
            self.row_counter = 0;
            self.column_counter += 1;
        } else {
            // keep incrementing through this column
            self.row_counter += 1;
        }

        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_columns = self.matrix.view_columns() - self.column_counter;
        match remaining_columns {
            0 => (0, Some(0)),
            1 => {
                // we're on the last column, so return how many items are left for us to
                // go through with the row counter
                let remaining_rows = self.matrix.view_rows() - self.row_counter;
                (remaining_rows, Some(remaining_rows))
            }
            x => {
                // we still have at least one full column left in addition to what's left
                // for this column's row counter
                let remaining_rows = self.matrix.view_rows() - self.row_counter;
                // each full column takes as many iterations as the matrix has rows
                let remaining_full_columns = (x - 1) * self.matrix.view_rows();
                let remaining = remaining_rows + remaining_full_columns;
                (remaining, Some(remaining))
            }
        }
    }
}

impl <'a, T: Clone, S: MatrixRef<T>> FusedIterator for ColumnMajorIterator<'a, T, S> {}
impl <'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for ColumnMajorIterator<'a, T, S> {}

/**
 * A row major iterator over all values in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as 1, 2, 3, 4
 */
#[derive(Debug)]
pub struct RowMajorIterator<'a, T: Clone, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column_counter: Column,
    row_counter: Row,
    finished: bool,
    _type: PhantomData<&'a T>,
}

impl <'a, T: Clone> RowMajorIterator<'a, T> {
    /**
     * Constructs a row major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> RowMajorIterator<T> {
        assert!(matrix.index_is_valid(0, 0), "Expected ({},{}) to be in range", 0, 0);
        RowMajorIterator {
            matrix,
            column_counter: 0,
            row_counter: 0,
            finished: false,
            _type: PhantomData,
        }
    }
}

impl <'a, T: Clone, S: MatrixRef<T>> Iterator for RowMajorIterator<'a, T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = unsafe {
            // Safety: We checked on creation that 0,0 is in range, and after getting
            // our next value we check if we hit the end of the matrix and will avoid
            // calling this on our next loop if we finished. Hence if the view_size has
            // not changed this read is in bounds and if they are able to be changed,
            // then the MatrixRef implementation is required to bounds check for us.
            Some(self.matrix.get_reference_unchecked(self.row_counter, self.column_counter).clone())
        };

        if self.column_counter == self.matrix.view_columns() - 1
                && self.row_counter == self.matrix.view_rows() -1 {
            // reached end of matrix for next iteration
            self.finished = true;
        }

        if self.column_counter == self.matrix.view_columns() - 1 {
            // reached end of a row, need to reset to first element in next row
            self.column_counter = 0;
            self.row_counter += 1;
        } else {
            // keep incrementing through this row
            self.column_counter += 1;
        }

        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_rows = self.matrix.view_rows() - self.row_counter;
        match remaining_rows {
            0 => (0, Some(0)),
            1 => {
                // we're on the last row, so return how many items are left for us to
                // go through with the column counter
                let remaining_columns = self.matrix.view_columns() - self.column_counter;
                (remaining_columns, Some(remaining_columns))
            }
            x => {
                // we still have at least one full row left in addition to what's left
                // for this row's column counter
                let remaining_columns = self.matrix.view_columns() - self.column_counter;
                // each full row takes as many iterations as the matrix has columns
                let remaining_full_rows = (x - 1) * self.matrix.view_columns();
                let remaining = remaining_columns + remaining_full_rows;
                (remaining, Some(remaining))
            }
        }
    }
}

impl <'a, T: Clone, S: MatrixRef<T>> FusedIterator for RowMajorIterator<'a, T, S> {}
impl <'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for RowMajorIterator<'a, T, S> {}

/**
 * An iterator over references to a column in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * Depending on the row iterator you want to obtain,
 * can either iterate through &1, &3 or &2, &4.
 */
#[derive(Debug)]
pub struct ColumnReferenceIterator<'a, T, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column: Column,
    range: Range<Row>,
    _type: PhantomData<&'a T>,
}

impl <'a, T> ColumnReferenceIterator<'a, T> {
    /**
     * Constructs a column iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>, column: Column) -> ColumnReferenceIterator<T> {
        assert!(matrix.index_is_valid(0, column), "Expected ({},{}) to be in range", 0, column);
        ColumnReferenceIterator {
            matrix,
            column,
            range: 0..matrix.view_rows(),
            _type: PhantomData,
        }
    }
}

impl <'a, T, S: MatrixRef<T>> Iterator for ColumnReferenceIterator<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(row) => unsafe {
                // Safety: We initialised the range to 0..matrix.view_rows(), and
                // checked we can read from this column at creation, hence if
                // the view_rows have not changed this read is in bounds and if
                // they are able to be changed, then the MatrixRef implementation is
                // required to bounds check for us.
                Some(self.matrix.get_reference_unchecked(row, self.column))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl <'a, T, S: MatrixRef<T>> FusedIterator for ColumnReferenceIterator<'a, T, S> {}
impl <'a, T, S: MatrixRef<T>> ExactSizeIterator for ColumnReferenceIterator<'a, T, S> {}

/**
 * An iterator over references to a row in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * Depending on the row iterator you want to obtain,
 * can either iterate through &1, &2 or &3, &4.
 */
#[derive(Debug)]
pub struct RowReferenceIterator<'a, T, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    row: Row,
    range: Range<Column>,
    _type: PhantomData<&'a T>,
}

impl <'a, T> RowReferenceIterator<'a, T> {
    /**
     * Constructs a row iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>, row: Row) -> RowReferenceIterator<T> {
        assert!(matrix.index_is_valid(row, 0), "Expected ({},{}) to be in range", row, 0);
        RowReferenceIterator {
            matrix,
            row,
            range: 0..matrix.view_columns(),
            _type: PhantomData,
        }
    }
}

impl <'a, T, S: MatrixRef<T>> Iterator for RowReferenceIterator<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(column) => unsafe {
                // Safety: We initialised the range to 0..matrix.view_columns(), and
                // checked we can read from this row at creation, hence if
                // the view_columns have not changed this read is in bounds and if
                // they are able to be changed, then the MatrixRef implementation is
                // required to bounds check for us.
                Some(self.matrix.get_reference_unchecked(self.row, column).clone())
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl <'a, T, S: MatrixRef<T>> FusedIterator for RowReferenceIterator<'a, T, S> {}
impl <'a, T, S: MatrixRef<T>> ExactSizeIterator for RowReferenceIterator<'a, T, S> {}

/**
 * A column major iterator over references to all values in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as &1, &3, &2, &4
 */
#[derive(Debug)]
pub struct ColumnMajorReferenceIterator<'a, T, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column_counter: Column,
    row_counter: Row,
    finished: bool,
    _type: PhantomData<&'a T>,
}

impl <'a, T> ColumnMajorReferenceIterator<'a, T> {
    /**
     * Constructs a column major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> ColumnMajorReferenceIterator<T> {
        assert!(matrix.index_is_valid(0, 0), "Expected ({},{}) to be in range", 0, 0);
        ColumnMajorReferenceIterator {
            matrix,
            column_counter: 0,
            row_counter: 0,
            finished: false,
            _type: PhantomData,
        }
    }
}

impl <'a, T, S: MatrixRef<T>> Iterator for ColumnMajorReferenceIterator<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = unsafe {
            // Safety: We checked on creation that 0,0 is in range, and after getting
            // our next value we check if we hit the end of the matrix and will avoid
            // calling this on our next loop if we finished. Hence if the view_size has
            // not changed this read is in bounds and if they are able to be changed,
            // then the MatrixRef implementation is required to bounds check for us.
            Some(self.matrix.get_reference_unchecked(self.row_counter, self.column_counter))
        };

        if self.row_counter == self.matrix.view_rows() - 1
                && self.column_counter == self.matrix.view_columns() -1 {
            // reached end of matrix for next iteration
            self.finished = true;
        }

        if self.row_counter == self.matrix.view_rows() - 1 {
            // reached end of a column, need to reset to first element in next column
            self.row_counter = 0;
            self.column_counter += 1;
        } else {
            // keep incrementing through this column
            self.row_counter += 1;
        }

        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_columns = self.matrix.view_columns() - self.column_counter;
        match remaining_columns {
            0 => (0, Some(0)),
            1 => {
                // we're on the last column, so return how many items are left for us to
                // go through with the row counter
                let remaining_rows = self.matrix.view_rows() - self.row_counter;
                (remaining_rows, Some(remaining_rows))
            }
            x => {
                // we still have at least one full column left in addition to what's left
                // for this column's row counter
                let remaining_rows = self.matrix.view_rows() - self.row_counter;
                // each full column takes as many iterations as the matrix has rows
                let remaining_full_columns = (x - 1) * self.matrix.view_rows();
                let remaining = remaining_rows + remaining_full_columns;
                (remaining, Some(remaining))
            }
        }
    }
}

impl <'a, T, S: MatrixRef<T>> FusedIterator for ColumnMajorReferenceIterator<'a, T, S> {}
impl <'a, T, S: MatrixRef<T>> ExactSizeIterator for ColumnMajorReferenceIterator<'a, T, S> {}

/**
 * A row major iterator over references to all values in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as &1, &2, &3, &4
 */
#[derive(Debug)]
pub struct RowMajorReferenceIterator<'a, T, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    column_counter: Column,
    row_counter: Row,
    finished: bool,
    _type: PhantomData<&'a T>,
}

impl <'a, T> RowMajorReferenceIterator<'a, T> {
    /**
     * Constructs a column major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> RowMajorReferenceIterator<T> {
        assert!(matrix.index_is_valid(0, 0), "Expected ({},{}) to be in range", 0, 0);
        RowMajorReferenceIterator {
            matrix,
            column_counter: 0,
            row_counter: 0,
            finished: false,
            _type: PhantomData,
        }
    }
}

impl <'a, T, S: MatrixRef<T>> Iterator for RowMajorReferenceIterator<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = unsafe {
            // Safety: We checked on creation that 0,0 is in range, and after getting
            // our next value we check if we hit the end of the matrix and will avoid
            // calling this on our next loop if we finished. Hence if the view_size has
            // not changed this read is in bounds and if they are able to be changed,
            // then the MatrixRef implementation is required to bounds check for us.
            Some(self.matrix.get_reference_unchecked(self.row_counter, self.column_counter))
        };

        if self.column_counter == self.matrix.view_columns() - 1
                && self.row_counter == self.matrix.view_rows() -1 {
            // reached end of matrix for next iteration
            self.finished = true;
        }

        if self.column_counter == self.matrix.view_columns() - 1 {
            // reached end of a row, need to reset to first element in next row
            self.column_counter = 0;
            self.row_counter += 1;
        } else {
            // keep incrementing through this row
            self.column_counter += 1;
        }

        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_rows = self.matrix.view_rows() - self.row_counter;
        match remaining_rows {
            0 => (0, Some(0)),
            1 => {
                // we're on the last row, so return how many items are left for us to
                // go through with the column counter
                let remaining_columns = self.matrix.view_columns() - self.column_counter;
                (remaining_columns, Some(remaining_columns))
            }
            x => {
                // we still have at least one full row left in addition to what's left
                // for this row's column counter
                let remaining_columns = self.matrix.view_columns() - self.column_counter;
                // each full row takes as many iterations as the matrix has columns
                let remaining_full_rows = (x - 1) * self.matrix.view_columns();
                let remaining = remaining_columns + remaining_full_rows;
                (remaining, Some(remaining))
            }
        }
    }
}

impl <'a, T, S: MatrixRef<T>> FusedIterator for RowMajorReferenceIterator<'a, T, S> {}
impl <'a, T, S: MatrixRef<T>> ExactSizeIterator for RowMajorReferenceIterator<'a, T, S> {}

/**
 * An iterator over the main diagonal in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as 1, 4
 *
 * If the matrix is not square this will stop at whichever row/colum is shorter.
 */
#[derive(Debug)]
pub struct DiagonalIterator<'a, T: Clone, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    range: Range<usize>,
    _type: PhantomData<&'a T>,
}

impl <'a, T: Clone> DiagonalIterator<'a, T> {
    /**
     * Constructs a diagonal iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> DiagonalIterator<T> {
        assert!(matrix.index_is_valid(0, 0), "Expected ({},{}) to be in range", 0, 0);
        DiagonalIterator {
            matrix,
            range: 0..std::cmp::min(matrix.view_rows(), matrix.view_columns()),
            _type: PhantomData,
        }
    }
}

impl <'a, T: Clone, S: MatrixRef<T>> Iterator for DiagonalIterator<'a, T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(i) => unsafe {
                // Safety: We initialised the range to 0..min(rows/columns), and
                // checked we can read from 0,0 at creation, hence if the view_size
                // has not changed this read is in bounds and if they are able to
                // be changed, then the MatrixRef implementation is required to bounds
                // check for us.
                Some(self.matrix.get_reference_unchecked(i, i).clone())
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl <'a, T: Clone, S: MatrixRef<T>> FusedIterator for DiagonalIterator<'a, T, S> {}
impl <'a, T: Clone, S: MatrixRef<T>> ExactSizeIterator for DiagonalIterator<'a, T, S> {}

/**
 * An iterator over references to the main diagonal in a matrix.
 *
 * For a 2x2 matrix such as `[ 1, 2; 3, 4]`: ie
 * ```ignore
 * [
 *   1, 2
 *   3, 4
 * ]
 * ```
 * The elements will be iterated through as &1, &4
 *
 * If the matrix is not square this will stop at whichever row/colum is shorter.
 */
#[derive(Debug)]
pub struct DiagonalReferenceIterator<'a, T, S: MatrixRef<T> = Matrix<T>> {
    matrix: &'a S,
    range: Range<usize>,
    _type: PhantomData<&'a T>,
}

impl <'a, T> DiagonalReferenceIterator<'a, T> {
    /**
     * Constructs a diagonal iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> DiagonalReferenceIterator<T> {
        assert!(matrix.index_is_valid(0, 0), "Expected ({},{}) to be in range", 0, 0);
        DiagonalReferenceIterator {
            matrix,
            range: 0..std::cmp::min(matrix.view_rows(), matrix.view_columns()),
            _type: PhantomData,
        }
    }
}

impl <'a, T, S: MatrixRef<T>> Iterator for DiagonalReferenceIterator<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.range.next() {
            None => None,
            Some(i) => unsafe {
                // Safety: We initialised the range to 0..min(rows/columns), and
                // checked we can read from 0,0 at creation, hence if the view_size
                // has not changed this read is in bounds and if they are able to
                // be changed, then the MatrixRef implementation is required to bounds
                // check for us.
                Some(self.matrix.get_reference_unchecked(i, i).clone())
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl <'a, T, S: MatrixRef<T>> FusedIterator for DiagonalReferenceIterator<'a, T, S> {}
impl <'a, T, S: MatrixRef<T>> ExactSizeIterator for DiagonalReferenceIterator<'a, T, S> {}
