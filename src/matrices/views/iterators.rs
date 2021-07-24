use std::iter::{ExactSizeIterator, FusedIterator};

use crate::matrices::{Row, Column};
use crate::matrices::views::{MatrixView, MatrixRef};

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
struct ColumnViewIterator<T, S> {
    matrix: MatrixView<T, S>,
    column: Column,
    counter: usize,
    finished: bool,
}

impl <T: Clone, S: MatrixRef<T>> ColumnViewIterator<T, S> {
    /**
     * Constructs a column iterator over this matrix.
     */
    pub fn new(matrix: MatrixView<T, S>, column: Column) -> ColumnViewIterator<T, S> {
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
