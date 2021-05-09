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

use crate::matrices::{Matrix, Row, Column};

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
pub struct ColumnIterator<'a, T: Clone> {
    matrix: &'a Matrix<T>,
    column: Column,
    counter: usize,
    finished: bool,
}

impl <'a, T: Clone> ColumnIterator<'a, T> {
    /**
     * Constructs a column iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>, column: Column) -> ColumnIterator<T> {
        ColumnIterator {
            matrix,
            column,
            counter: 0,
            finished: false,
        }
    }
}

impl <'a, T: Clone> Iterator for ColumnIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = Some(self.matrix.get(self.counter, self.column));

        if self.counter == self.matrix.rows() - 1 {
            self.finished = true;
        }

        self.counter += 1;

        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.matrix.rows() - self.counter;
        (remaining, Some(remaining))
    }
}

impl <'a, T: Clone> FusedIterator for ColumnIterator<'a, T> {}
impl <'a, T: Clone> ExactSizeIterator for ColumnIterator<'a, T> {}

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
pub struct RowIterator<'a, T: Clone> {
    matrix: &'a Matrix<T>,
    row: Row,
    counter: usize,
    finished: bool,
}

impl <'a, T: Clone> RowIterator<'a, T> {
    /**
     * Constructs a row iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>, row: Row) -> RowIterator<T> {
        RowIterator {
            matrix,
            row,
            counter: 0,
            finished: false,
        }
    }
}

impl <'a, T: Clone> Iterator for RowIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = Some(self.matrix.get(self.row, self.counter));

        if self.counter == self.matrix.columns() - 1 {
            self.finished = true;
        }

        self.counter += 1;

        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.matrix.columns() - self.counter;
        (remaining, Some(remaining))
    }
}

impl <'a, T: Clone> FusedIterator for RowIterator<'a, T> {}
impl <'a, T: Clone> ExactSizeIterator for RowIterator<'a, T> {}

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
pub struct ColumnMajorIterator<'a, T: Clone> {
    matrix: &'a Matrix<T>,
    column_counter: Column,
    row_counter: Row,
    finished: bool,
}

impl <'a, T: Clone> ColumnMajorIterator<'a, T> {
    /**
     * Constructs a column major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> ColumnMajorIterator<T> {
        ColumnMajorIterator {
            matrix,
            column_counter: 0,
            row_counter: 0,
            finished: false,
        }
    }
}

impl <'a, T: Clone> Iterator for ColumnMajorIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = Some(self.matrix.get(self.row_counter, self.column_counter));

        if self.row_counter == self.matrix.rows() - 1
                && self.column_counter == self.matrix.columns() -1 {
            // reached end of matrix for next iteration
            self.finished = true;
        }

        if self.row_counter == self.matrix.rows() - 1 {
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
        let remaining_columns = self.matrix.columns() - self.column_counter;
        match remaining_columns {
            0 => (0, Some(0)),
            1 => {
                // we're on the last column, so return how many items are left for us to
                // go through with the row counter
                let remaining_rows = self.matrix.rows() - self.row_counter;
                (remaining_rows, Some(remaining_rows))
            }
            x => {
                // we still have at least one full column left in addition to what's left
                // for this column's row counter
                let remaining_rows = self.matrix.rows() - self.row_counter;
                // each full column takes as many iterations as the matrix has rows
                let remaining_full_columns = (x - 1) * self.matrix.rows();
                let remaining = remaining_rows + remaining_full_columns;
                (remaining, Some(remaining))
            }
        }
    }
}

impl <'a, T: Clone> FusedIterator for ColumnMajorIterator<'a, T> {}
impl <'a, T: Clone> ExactSizeIterator for ColumnMajorIterator<'a, T> {}

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
pub struct RowMajorIterator<'a, T: Clone> {
    matrix: &'a Matrix<T>,
    column_counter: Column,
    row_counter: Row,
    finished: bool,
}

impl <'a, T: Clone> RowMajorIterator<'a, T> {
    /**
     * Constructs a row major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> RowMajorIterator<T> {
        RowMajorIterator {
            matrix,
            column_counter: 0,
            row_counter: 0,
            finished: false,
        }
    }
}

impl <'a, T: Clone> Iterator for RowMajorIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = Some(self.matrix.get(self.row_counter, self.column_counter));

        if self.column_counter == self.matrix.columns() - 1
                && self.row_counter == self.matrix.rows() -1 {
            // reached end of matrix for next iteration
            self.finished = true;
        }

        if self.column_counter == self.matrix.columns() - 1 {
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
        let remaining_rows = self.matrix.rows() - self.row_counter;
        match remaining_rows {
            0 => (0, Some(0)),
            1 => {
                // we're on the last row, so return how many items are left for us to
                // go through with the column counter
                let remaining_columns = self.matrix.columns() - self.column_counter;
                (remaining_columns, Some(remaining_columns))
            }
            x => {
                // we still have at least one full row left in addition to what's left
                // for this row's column counter
                let remaining_columns = self.matrix.columns() - self.column_counter;
                // each full row takes as many iterations as the matrix has columns
                let remaining_full_rows = (x - 1) * self.matrix.columns();
                let remaining = remaining_columns + remaining_full_rows;
                (remaining, Some(remaining))
            }
        }
    }
}

impl <'a, T: Clone> FusedIterator for RowMajorIterator<'a, T> {}
impl <'a, T: Clone> ExactSizeIterator for RowMajorIterator<'a, T> {}

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
pub struct ColumnReferenceIterator<'a, T> {
    matrix: &'a Matrix<T>,
    column: Column,
    counter: usize,
    finished: bool,
}

impl <'a, T> ColumnReferenceIterator<'a, T> {
    /**
     * Constructs a column iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>, column: Column) -> ColumnReferenceIterator<T> {
        ColumnReferenceIterator {
            matrix,
            column,
            counter: 0,
            finished: false,
        }
    }
}

impl <'a, T> Iterator for ColumnReferenceIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = Some(self.matrix.get_reference(self.counter, self.column));

        if self.counter == self.matrix.rows() - 1 {
            self.finished = true;
        }

        self.counter += 1;

        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.matrix.rows() - self.counter;
        (remaining, Some(remaining))
    }
}

impl <'a, T> FusedIterator for ColumnReferenceIterator<'a, T> {}
impl <'a, T> ExactSizeIterator for ColumnReferenceIterator<'a, T> {}

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
pub struct RowReferenceIterator<'a, T> {
    matrix: &'a Matrix<T>,
    row: Row,
    counter: usize,
    finished: bool,
}

impl <'a, T> RowReferenceIterator<'a, T> {
    /**
     * Constructs a row iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>, row: Row) -> RowReferenceIterator<T> {
        RowReferenceIterator {
            matrix,
            row,
            counter: 0,
            finished: false,
        }
    }
}

impl <'a, T> Iterator for RowReferenceIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = Some(self.matrix.get_reference(self.row, self.counter));

        if self.counter == self.matrix.columns() - 1 {
            self.finished = true;
        }

        self.counter += 1;

        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.matrix.columns() - self.counter;
        (remaining, Some(remaining))
    }
}

impl <'a, T> FusedIterator for RowReferenceIterator<'a, T> {}
impl <'a, T> ExactSizeIterator for RowReferenceIterator<'a, T> {}

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
pub struct ColumnMajorReferenceIterator<'a, T> {
    matrix: &'a Matrix<T>,
    column_counter: Column,
    row_counter: Row,
    finished: bool,
}

impl <'a, T> ColumnMajorReferenceIterator<'a, T> {
    /**
     * Constructs a column major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> ColumnMajorReferenceIterator<T> {
        ColumnMajorReferenceIterator {
            matrix,
            column_counter: 0,
            row_counter: 0,
            finished: false,
        }
    }
}

impl <'a, T> Iterator for ColumnMajorReferenceIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = Some(self.matrix.get_reference(self.row_counter, self.column_counter));

        if self.row_counter == self.matrix.rows() - 1
                && self.column_counter == self.matrix.columns() -1 {
            // reached end of matrix for next iteration
            self.finished = true;
        }

        if self.row_counter == self.matrix.rows() - 1 {
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
        let remaining_columns = self.matrix.columns() - self.column_counter;
        match remaining_columns {
            0 => (0, Some(0)),
            1 => {
                // we're on the last column, so return how many items are left for us to
                // go through with the row counter
                let remaining_rows = self.matrix.rows() - self.row_counter;
                (remaining_rows, Some(remaining_rows))
            }
            x => {
                // we still have at least one full column left in addition to what's left
                // for this column's row counter
                let remaining_rows = self.matrix.rows() - self.row_counter;
                // each full column takes as many iterations as the matrix has rows
                let remaining_full_columns = (x - 1) * self.matrix.rows();
                let remaining = remaining_rows + remaining_full_columns;
                (remaining, Some(remaining))
            }
        }
    }
}

impl <'a, T> FusedIterator for ColumnMajorReferenceIterator<'a, T> {}
impl <'a, T> ExactSizeIterator for ColumnMajorReferenceIterator<'a, T> {}

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
    pub struct RowMajorReferenceIterator<'a, T> {
    matrix: &'a Matrix<T>,
    column_counter: Column,
    row_counter: Row,
    finished: bool,
}

impl <'a, T> RowMajorReferenceIterator<'a, T> {
    /**
     * Constructs a column major iterator over this matrix.
     */
    pub fn new(matrix: &Matrix<T>) -> RowMajorReferenceIterator<T> {
        RowMajorReferenceIterator {
            matrix,
            column_counter: 0,
            row_counter: 0,
            finished: false,
        }
    }
}

impl <'a, T> Iterator for RowMajorReferenceIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = Some(self.matrix.get_reference(self.row_counter, self.column_counter));

        if self.column_counter == self.matrix.columns() - 1
                && self.row_counter == self.matrix.rows() -1 {
            // reached end of matrix for next iteration
            self.finished = true;
        }

        if self.column_counter == self.matrix.columns() - 1 {
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
        let remaining_rows = self.matrix.rows() - self.row_counter;
        match remaining_rows {
            0 => (0, Some(0)),
            1 => {
                // we're on the last row, so return how many items are left for us to
                // go through with the column counter
                let remaining_columns = self.matrix.columns() - self.column_counter;
                (remaining_columns, Some(remaining_columns))
            }
            x => {
                // we still have at least one full row left in addition to what's left
                // for this row's column counter
                let remaining_columns = self.matrix.columns() - self.column_counter;
                // each full row takes as many iterations as the matrix has columns
                let remaining_full_rows = (x - 1) * self.matrix.columns();
                let remaining = remaining_columns + remaining_full_rows;
                (remaining, Some(remaining))
            }
        }
    }
}

impl <'a, T> FusedIterator for RowMajorReferenceIterator<'a, T> {}
impl <'a, T> ExactSizeIterator for RowMajorReferenceIterator<'a, T> {}
