/*!
 * Iterators over parts of a Matrix
 */

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
 * Depending on the row iterator you want to obtain,
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
}

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
}

/**
 * An column major iterator over all values in a matrix.
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

        if self.row_counter == self.matrix.rows() - 1 && self.column_counter == self.matrix.columns() -1 {
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
}
