use std::iter::{ExactSizeIterator, FusedIterator};

use crate::matrices::{Row, Column};
use crate::matrices::views::{MatrixView, MatrixRef};

/**
 * An iterator over a column in a matrix view.
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
struct ColumnViewIterator<T, S> {
    matrix: MatrixView<T, S>,
    column: Column,
    counter: usize,
    finished: bool,
}

impl <T: Clone, S: MatrixRef<T>> ColumnViewIterator<T, S> {
    /**
     * Constructs a column iterator over this matrix view.
     */
    fn new(matrix: MatrixView<T, S>, column: Column) -> ColumnViewIterator<T, S> {
        ColumnViewIterator {
            matrix,
            column,
            counter: 0,
            finished: false,
        }
    }
}

impl <T: Clone, S: MatrixRef<T>> Iterator for ColumnViewIterator<T, S> {
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

impl <T: Clone, S: MatrixRef<T>> FusedIterator for ColumnViewIterator<T, S> {}
impl <T: Clone, S: MatrixRef<T>> ExactSizeIterator for ColumnViewIterator<T, S> {}

/**
 * An iterator over a row in a matrix view.
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
struct RowViewIterator<T, S> {
    matrix: MatrixView<T, S>,
    row: Row,
    counter: usize,
    finished: bool,
}

impl <T: Clone, S: MatrixRef<T>> RowViewIterator<T, S> {
    /**
     * Constructs a row iterator over this matrix view.
     */
    fn new(matrix: MatrixView<T, S>, row: Row) -> RowViewIterator<T, S> {
        RowViewIterator {
            matrix,
            row,
            counter: 0,
            finished: false,
        }
    }
}

impl <T: Clone, S: MatrixRef<T>> Iterator for RowViewIterator<T, S> {
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

impl <T: Clone, S: MatrixRef<T>> FusedIterator for RowViewIterator<T, S> {}
impl <T: Clone, S: MatrixRef<T>> ExactSizeIterator for RowViewIterator<T, S> {}

/**
 * A column major iterator over all values in a matrix view.
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
struct ColumnMajorViewIterator<T, S> {
    matrix: MatrixView<T, S>,
    column_counter: Column,
    row_counter: Row,
    finished: bool,
}

impl <T: Clone, S: MatrixRef<T>> ColumnMajorViewIterator<T, S> {
    /**
     * Constructs a column major iterator over this matrix view.
     */
    fn new(matrix: MatrixView<T, S>) -> ColumnMajorViewIterator<T, S> {
        ColumnMajorViewIterator {
            matrix,
            column_counter: 0,
            row_counter: 0,
            finished: false,
        }
    }
}

impl <T: Clone, S: MatrixRef<T>> Iterator for ColumnMajorViewIterator<T, S> {
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

impl <T: Clone, S: MatrixRef<T>> FusedIterator for ColumnMajorViewIterator<T, S> {}
impl <T: Clone, S: MatrixRef<T>> ExactSizeIterator for ColumnMajorViewIterator<T, S> {}

/**
 * A row major iterator over all values in a matrix view.
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
struct RowMajorViewIterator<T, S> {
    matrix: MatrixView<T, S>,
    column_counter: Column,
    row_counter: Row,
    finished: bool,
}

impl <T: Clone, S: MatrixRef<T>> RowMajorViewIterator<T, S> {
    /**
     * Constructs a row major iterator over this matrix view.
     */
    fn new(matrix: MatrixView<T, S>) -> RowMajorViewIterator<T, S> {
        RowMajorViewIterator {
            matrix,
            column_counter: 0,
            row_counter: 0,
            finished: false,
        }
    }
}

impl <T: Clone, S: MatrixRef<T>> Iterator for RowMajorViewIterator<T, S> {
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

impl <T: Clone, S: MatrixRef<T>> FusedIterator for RowMajorViewIterator<T, S> {}
impl <T: Clone, S: MatrixRef<T>> ExactSizeIterator for RowMajorViewIterator<T, S> {}

/**
 * An iterator over the main diagonal in a matrix view.
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
struct DiagonalViewIterator<T, S> {
    matrix: MatrixView<T, S>,
    counter: usize,
    final_index: usize,
    finished: bool,
}

impl <T: Clone, S: MatrixRef<T>> DiagonalViewIterator<T, S> {
    /**
     * Constructs a diagonal iterator over this matrix view.
     */
    fn new(matrix: MatrixView<T, S>) -> DiagonalViewIterator<T, S> {
        DiagonalViewIterator {
            final_index: std::cmp::min(matrix.rows(), matrix.columns()),
            matrix,
            counter: 0,
            finished: false,
        }
    }
}

impl <T: Clone, S: MatrixRef<T>> Iterator for DiagonalViewIterator<T, S> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None
        }

        let value = Some(self.matrix.get(self.counter, self.counter));

        if self.counter == self.final_index - 1 {
            self.finished = true;
        }

        self.counter += 1;

        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.final_index - self.counter;
        (remaining, Some(remaining))
    }
}

impl <T: Clone, S: MatrixRef<T>> FusedIterator for DiagonalViewIterator<T, S> {}
impl <T: Clone, S: MatrixRef<T>> ExactSizeIterator for DiagonalViewIterator<T, S> {}
