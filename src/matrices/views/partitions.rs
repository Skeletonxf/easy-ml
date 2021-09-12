use crate::matrices::{Row, Column};
#[allow(unused)] // used in doc links
use crate::matrices::Matrix;
use crate::matrices::views::{DataLayout, MatrixView, MatrixRef, MatrixMut};

/**
 * A mutably borrowed part of a matrix.
 *
 * Rust's borrow checker does not not permit overlapping exclusive references, so you cannot
 * simply construct multiple views into a [Matrix](Matrix) by creating each one sequentially as
 * you can for immutable/shared references to a Matrix.
 *
 * ```
 * use easy_ml::matrices::Matrix;
 * use easy_ml::matrices::views::MatrixRange;
 * let matrix = Matrix::row(vec![1, 2, 3]);
 * let one = MatrixRange::from(&matrix, 0..1, 0..1);
 * let two = MatrixRange::from(&matrix, 0..1, 1..2);
 * let three = MatrixRange::from(&matrix, 0..1, 2..3);
 * let four = MatrixRange::from(&matrix, 0..1, 0..3);
 * ```
 *
 * MatrixPart instead holds only a mutable reference to a slice into a Matrix's buffer. It does
 * not borrow the entire Matrix, and thus is used as the container for Matrix APIs which partition
 * a Matrix into multiple non overlapping parts. The Matrix can then be independently mutated
 * by each of the MatrixParts.
 *
 * See
 * - [`Matrix::partition_quadrants`](Matrix::partition_quadrants)
 */
#[derive(Debug)]
pub struct MatrixPart<'source, T> {
    data: Vec<&'source mut [T]>,
    rows: Row,
    columns: Column,
}

impl <'a, T> MatrixPart<'a, T> {
    pub(crate) fn new(data: Vec<&'a mut [T]>, rows: Row, columns: Column) -> MatrixPart<'a, T> {
        MatrixPart {
            data,
            rows,
            columns
        }
    }
}

// # Safety
//
// We don't implement interior mutability and we can't be resized anyway since our
// buffer is not owned.
/**
 * A MatrixPart implements MatrixRef.
 */
unsafe impl <'a, T> MatrixRef<T> for MatrixPart<'a, T> {
    fn try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        Some(&self.data[row][column])
    }

    fn view_rows(&self) -> Row {
        self.rows
    }

    fn view_columns(&self) -> Column {
        self.columns
    }

    unsafe fn get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        self.data.get_unchecked(row).get_unchecked(column)
    }

    fn data_layout(&self) -> DataLayout {
        DataLayout::RowMajor
    }
}

// # Safety
//
// We don't implement interior mutability and we can't be resized anyway since our
// buffer is not owned.
/**
 * A MatrixPart implements MatrixMut.
 */
unsafe impl <'a, T> MatrixMut<T> for MatrixPart<'a, T> {
    fn try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        Some(&mut self.data[row][column])
    }

    unsafe fn get_reference_unchecked_mut(&mut self, row: Row, column: Column) -> &mut T {
        self.data.get_unchecked_mut(row).get_unchecked_mut(column)
    }
}

#[derive(Debug)]
pub struct MatrixQuadrants<'source, T> {
    pub top_left: MatrixView<T, MatrixPart<'source, T>>,
    pub top_right: MatrixView<T, MatrixPart<'source, T>>,
    pub bottom_left: MatrixView<T, MatrixPart<'source, T>>,
    pub bottom_right: MatrixView<T, MatrixPart<'source, T>>,
}

impl <'a, T> std::fmt::Display for MatrixQuadrants<'a, T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Top Left:\n{}\nTop Right:\n{}\nBottom Left:\n{}\nBottom Right:\n{}\n",
            self.top_left, self.top_right, self.bottom_left, self.bottom_right
        )
    }
}
