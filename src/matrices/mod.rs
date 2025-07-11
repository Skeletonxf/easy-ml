/*!
 * Generic matrix type.
 *
 * Matrices are generic over some type `T`. If `T` is [Numeric](super::numeric) then
 * the matrix can be used in a mathematical way.
 */

#[cfg(feature = "serde")]
use serde::Serialize;

mod errors;
pub mod iterators;
pub mod operations;
pub mod slices;
pub mod views;

pub use errors::ScalarConversionError;

use crate::linear_algebra;
use crate::matrices::iterators::*;
use crate::matrices::slices::Slice2D;
use crate::matrices::views::{
    IndexRange, MatrixMask, MatrixPart, MatrixQuadrants, MatrixRange, MatrixReverse, MatrixView,
    Reverse,
};
use crate::numeric::extra::{Real, RealRef};
use crate::numeric::{Numeric, NumericRef};

/**
 * A general purpose matrix of some type. This type may implement
 * no traits, in which case the matrix will be rather useless. If the
 * type implements [`Clone`] most storage and accessor methods are defined and if the type
 * implements [`Numeric`](super::numeric) then the matrix can be used in
 * a mathematical way.
 *
 * When doing numeric operations with Matrices you should be careful to not
 * consume a matrix by accidentally using it by value. All the operations are
 * also defined on references to matrices so you should favor `&x * &y` style
 * notation for matrices you intend to continue using. There are also convenience
 * operations defined for a matrix and a scalar.
 *
 * # Matrix size invariants
 *
 * Matrices must always be at least 1x1. You cannot construct a matrix with no rows or
 * no columns, and any function that resizes matrices will error if you try to use it
 * in a way that would construct a 0x1, 1x0, or 0x0 matrix. The maximum size of a matrix
 * is dependent on the platform's `std::isize::MAX` value. Matrices with dimensions NxM
 * such that N * M < `std::isize::MAX` should not cause any errors in this library, but
 * attempting to expand their size further may cause panics and or errors. At the time of
 * writing it is no longer possible to construct or use matrices where the product of their
 * number of rows and columns exceed `std::isize::MAX`, but some constructor methods may be used
 * to attempt this. Concerned readers should note that on a 64 bit computer this maximum
 * value is 9,223,372,036,854,775,807 so running out of memory is likely to occur first.
 *
 * # Matrix layout and iterator performance
 *
 * [See iterators submodule for Matrix layout and iterator performance](iterators#matrix-layout-and-iterator-performance)
 *
 * # Matrix operations
 *
 * [See operations submodule](operations)
 */
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: Row,
    columns: Column,
}

/// The maximum row and column lengths are usize, due to the internal storage being backed by Vec
pub type Row = usize;
/// The maximum row and column lengths are usize, due to the internal storage being backed by Vec
pub type Column = usize;

/**
 * Methods for matrices of any type, including non numerical types such as bool.
 */
impl<T> Matrix<T> {
    /**
     * Creates a 1x1 matrix from some scalar
     */
    pub fn from_scalar(value: T) -> Matrix<T> {
        Matrix {
            data: vec![value],
            rows: 1,
            columns: 1,
        }
    }

    /**
     * Creates a row vector (1xN) from a list
     *
     * # Panics
     *
     * Panics if no values are provided. Note: this method erroneously did not validate its inputs
     * in Easy ML versions up to and including 1.7.0
     */
    #[track_caller]
    pub fn row(values: Vec<T>) -> Matrix<T> {
        assert!(!values.is_empty(), "No values provided");
        Matrix {
            columns: values.len(),
            data: values,
            rows: 1,
        }
    }

    /**
     * Creates a column vector (Nx1) from a list
     *
     * # Panics
     *
     * Panics if no values are provided. Note: this method erroneously did not validate its inputs
     * in Easy ML versions up to and including 1.7.0
     */
    #[track_caller]
    pub fn column(values: Vec<T>) -> Matrix<T> {
        assert!(!values.is_empty(), "No values provided");
        Matrix {
            rows: values.len(),
            data: values,
            columns: 1,
        }
    }

    /**
     * Creates a matrix from a nested array of values, each inner vector
     * being a row, and hence the outer vector containing all rows in sequence, the
     * same way as when writing matrices in mathematics.
     *
     * Example of a 2 x 3 matrix in both notations:
     * ```ignore
     *   [
     *      1, 2, 4
     *      8, 9, 3
     *   ]
     * ```
     * ```
     * use easy_ml::matrices::Matrix;
     * Matrix::from(vec![
     *     vec![ 1, 2, 4 ],
     *     vec![ 8, 9, 3 ]
     * ]);
     * ```
     *
     * # Panics
     *
     * Panics if the input is jagged or rows or column length is 0.
     */
    #[track_caller]
    pub fn from(mut values: Vec<Vec<T>>) -> Matrix<T> {
        assert!(!values.is_empty(), "No rows defined");
        // check length of first row is > 1
        assert!(!values[0].is_empty(), "No column defined");
        // check length of each row is the same
        assert!(
            values.iter().map(|x| x.len()).all(|x| x == values[0].len()),
            "Inconsistent size"
        );
        // flatten the data into a row major layout
        let rows = values.len();
        let columns = values[0].len();
        let mut data = Vec::with_capacity(rows * columns);
        let mut value_stream = values.drain(..);
        for _ in 0..rows {
            let mut value_row_stream = value_stream.next().unwrap();
            let mut row_of_values = value_row_stream.drain(..);
            for _ in 0..columns {
                data.push(row_of_values.next().unwrap());
            }
        }
        Matrix {
            data,
            rows,
            columns,
        }
    }

    /**
     * Creates a matrix with the specified size from a row major vec of data.
     * The length of the vec must match the size of the matrix or the constructor
     * will panic.
     *
     * Example of a 2 x 3 matrix in both notations:
     * ```ignore
     *   [
     *      1, 2, 4
     *      8, 9, 3
     *   ]
     * ```
     * ```
     * use easy_ml::matrices::Matrix;
     * Matrix::from_flat_row_major((2, 3), vec![
     *     1, 2, 4,
     *     8, 9, 3
     * ]);
     * ```
     *
     * This method is more efficient than [`Matrix::from`](Matrix::from())
     * but requires specifying the size explicitly and manually keeping track of where rows
     * start and stop.
     *
     * # Panics
     *
     * Panics if the length of the vec does not match the size of the matrix, or no values are
     * provided. Note: this method erroneously did not validate its inputs were not empty in
     * Easy ML versions up to and including 1.7.0
     */
    #[track_caller]
    pub fn from_flat_row_major(size: (Row, Column), values: Vec<T>) -> Matrix<T> {
        assert!(
            size.0 * size.1 == values.len(),
            "Inconsistent size, attempted to construct a {}x{} matrix but provided with {} elements.",
            size.0,
            size.1,
            values.len()
        );
        assert!(!values.is_empty(), "No values provided");
        Matrix {
            data: values,
            rows: size.0,
            columns: size.1,
        }
    }

    /**
     * Creates a matrix with the specified size initalised from a function.
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * let matrix = Matrix::from_fn((4, 4), |(r, c)| r * c);
     * assert_eq!(
     *     matrix,
     *     Matrix::from(vec![
     *         vec![ 0, 0, 0, 0 ],
     *         vec![ 0, 1, 2, 3 ],
     *         vec![ 0, 2, 4, 6 ],
     *         vec![ 0, 3, 6, 9 ],
     *     ])
     * );
     * ```
     *
     * # Panics
     *
     * Panics if the size has 0 rows or columns.
     */
    #[track_caller]
    pub fn from_fn<F>(size: (Row, Column), mut producer: F) -> Matrix<T>
    where
        F: FnMut((Row, Column)) -> T,
    {
        use crate::tensors::indexing::ShapeIterator;
        let length = size.0 * size.1;
        let mut data = Vec::with_capacity(length);
        let iterator = ShapeIterator::from([("row", size.0), ("column", size.1)]);
        for [r, c] in iterator {
            data.push(producer((r, c)));
        }
        Matrix::from_flat_row_major(size, data)
    }

    #[deprecated(
        since = "1.1.0",
        note = "Incorrect use of terminology, a unit matrix is another term for an identity matrix, please use `from_scalar` instead"
    )]
    pub fn unit(value: T) -> Matrix<T> {
        Matrix::from_scalar(value)
    }

    /**
     * Returns the dimensionality of this matrix in Row, Column format
     */
    pub fn size(&self) -> (Row, Column) {
        (self.rows, self.columns)
    }

    /**
     * Gets the number of rows in this matrix.
     */
    pub fn rows(&self) -> Row {
        self.rows
    }

    /**
     * Gets the number of columns in this matrix.
     */
    pub fn columns(&self) -> Column {
        self.columns
    }

    /**
     * Matrix data is stored as row major, so each row is stored as
     * adjacent items going through the different columns. Therefore,
     * to index this flattened representation we jump down in row sized
     * blocks to reach the correct row, and then jump further equal to
     * the column. The confusing thing is that the number of columns
     * this matrix has is the length of each of the rows in this matrix,
     * and vice versa.
     */
    fn get_index(&self, row: Row, column: Column) -> usize {
        column + (row * self.columns())
    }

    /**
     * The reverse of [get_index], converts from the flattened storage
     * in memory into the row and column to index at this position.
     *
     * Matrix data is stored as row major, so each multiple of the number
     * of columns starts a new row, and each index modulo the columns
     * gives the column.
     */
    #[allow(dead_code)]
    fn get_row_column(&self, index: usize) -> (Row, Column) {
        (index / self.columns(), index % self.columns())
    }

    /**
     * Gets a reference to the value at this row and column. Rows and Columns are 0 indexed.
     *
     * # Panics
     *
     * Panics if the index is out of range.
     */
    #[track_caller]
    pub fn get_reference(&self, row: Row, column: Column) -> &T {
        assert!(row < self.rows(), "Row out of index");
        assert!(column < self.columns(), "Column out of index");
        &self.data[self.get_index(row, column)]
    }

    /**
     * Gets a mutable reference to the value at this row and column.
     * Rows and Columns are 0 indexed.
     *
     * # Panics
     *
     * Panics if the index is out of range.
     */
    #[track_caller]
    pub fn get_reference_mut(&mut self, row: Row, column: Column) -> &mut T {
        assert!(row < self.rows(), "Row out of index");
        assert!(column < self.columns(), "Column out of index");
        let index = self.get_index(row, column);
        // borrow for get_index ends
        &mut self.data[index]
    }

    /**
     * Not public API because don't want to name clash with the method on MatrixRef
     * that calls this.
     */
    pub(crate) fn _try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        if row < self.rows() && column < self.columns() {
            Some(&self.data[self.get_index(row, column)])
        } else {
            None
        }
    }

    /**
     * Not public API because don't want to name clash with the method on MatrixRef
     * that calls this.
     */
    pub(crate) unsafe fn _get_reference_unchecked(&self, row: Row, column: Column) -> &T {
        unsafe { self.data.get_unchecked(self.get_index(row, column)) }
    }

    /**
     * Sets a new value to this row and column. Rows and Columns are 0 indexed.
     *
     * # Panics
     *
     * Panics if the index is out of range.
     */
    #[track_caller]
    pub fn set(&mut self, row: Row, column: Column, value: T) {
        assert!(row < self.rows(), "Row out of index");
        assert!(column < self.columns(), "Column out of index");
        let index = self.get_index(row, column);
        // borrow for get_index ends
        self.data[index] = value;
    }

    /**
     * Not public API because don't want to name clash with the method on MatrixMut
     * that calls this.
     */
    pub(crate) fn _try_get_reference_mut(&mut self, row: Row, column: Column) -> Option<&mut T> {
        if row < self.rows() && column < self.columns() {
            let index = self.get_index(row, column);
            // borrow for get_index ends
            Some(&mut self.data[index])
        } else {
            None
        }
    }

    /**
     * Not public API because don't want to name clash with the method on MatrixMut
     * that calls this.
     */
    pub(crate) unsafe fn _get_reference_unchecked_mut(
        &mut self,
        row: Row,
        column: Column,
    ) -> &mut T {
        unsafe {
            let index = self.get_index(row, column);
            // borrow for get_index ends
            self.data.get_unchecked_mut(index)
        }
    }

    /**
     * Removes a row from this Matrix, shifting all other rows to the left.
     * Rows are 0 indexed.
     *
     * # Panics
     *
     * This will panic if the row does not exist or the matrix only has one row.
     */
    #[track_caller]
    pub fn remove_row(&mut self, row: Row) {
        assert!(self.rows() > 1);
        let mut r = 0;
        let mut c = 0;
        // drop the values at the specified row
        let columns = self.columns();
        self.data.retain(|_| {
            let keep = r != row;
            if c < (columns - 1) {
                c += 1;
            } else {
                r += 1;
                c = 0;
            }
            keep
        });
        self.rows -= 1;
    }

    /**
     * Removes a column from this Matrix, shifting all other columns to the left.
     * Columns are 0 indexed.
     *
     * # Panics
     *
     * This will panic if the column does not exist or the matrix only has one column.
     */
    #[track_caller]
    pub fn remove_column(&mut self, column: Column) {
        assert!(self.columns() > 1);
        let mut r = 0;
        let mut c = 0;
        // drop the values at the specified column
        let columns = self.columns();
        self.data.retain(|_| {
            let keep = c != column;
            if c < (columns - 1) {
                c += 1;
            } else {
                r += 1;
                c = 0;
            }
            keep
        });
        self.columns -= 1;
    }

    /**
     * Returns an iterator over references to a column vector in this matrix.
     * Columns are 0 indexed.
     *
     * # Panics
     *
     * Panics if the column does not exist in this matrix.
     */
    #[track_caller]
    pub fn column_reference_iter(&self, column: Column) -> ColumnReferenceIterator<T> {
        ColumnReferenceIterator::new(self, column)
    }

    /**
     * Returns an iterator over references to a row vector in this matrix.
     * Rows are 0 indexed.
     *
     * # Panics
     *
     * Panics if the row does not exist in this matrix.
     */
    #[track_caller]
    pub fn row_reference_iter(&self, row: Row) -> RowReferenceIterator<T> {
        RowReferenceIterator::new(self, row)
    }

    /**
     * Returns an iterator over mutable references to a column vector in this matrix.
     * Columns are 0 indexed.
     *
     * # Panics
     *
     * Panics if the column does not exist in this matrix.
     */
    #[track_caller]
    pub fn column_reference_mut_iter(&mut self, column: Column) -> ColumnReferenceMutIterator<T> {
        ColumnReferenceMutIterator::new(self, column)
    }

    /**
     * Returns an iterator over mutable references to a row vector in this matrix.
     * Rows are 0 indexed.
     *
     * # Panics
     *
     * Panics if the row does not exist in this matrix.
     */
    #[track_caller]
    pub fn row_reference_mut_iter(&mut self, row: Row) -> RowReferenceMutIterator<T> {
        RowReferenceMutIterator::new(self, row)
    }

    /**
     * Returns a column major iterator over references to all values in this matrix,
     * proceeding through each column in order.
     */
    pub fn column_major_reference_iter(&self) -> ColumnMajorReferenceIterator<T> {
        ColumnMajorReferenceIterator::new(self)
    }

    /**
     * Returns a row major iterator over references to all values in this matrix,
     * proceeding through each row in order.
     */
    pub fn row_major_reference_iter(&self) -> RowMajorReferenceIterator<T> {
        RowMajorReferenceIterator::new(self)
    }

    // Non public row major reference iterator since we don't want to expose our implementation
    // details to public API since then we could never change them.
    pub(crate) fn direct_row_major_reference_iter(&self) -> std::slice::Iter<T> {
        self.data.iter()
    }

    // Non public row major reference iterator since we don't want to expose our implementation
    // details to public API since then we could never change them.
    pub(crate) fn direct_row_major_reference_iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.data.iter_mut()
    }

    /**
     * Returns a column major iterator over mutable references to all values in this matrix,
     * proceeding through each column in order.
     */
    pub fn column_major_reference_mut_iter(&mut self) -> ColumnMajorReferenceMutIterator<T> {
        ColumnMajorReferenceMutIterator::new(self)
    }

    /**
     * Returns a row major iterator over mutable references to all values in this matrix,
     * proceeding through each row in order.
     */
    pub fn row_major_reference_mut_iter(&mut self) -> RowMajorReferenceMutIterator<T> {
        RowMajorReferenceMutIterator::new(self)
    }

    /**
     * Creates a column major iterator over all values in this matrix,
     * proceeding through each column in order.
     */
    pub fn column_major_owned_iter(self) -> ColumnMajorOwnedIterator<T>
    where
        T: Default,
    {
        ColumnMajorOwnedIterator::new(self)
    }

    /**
     * Creates a row major iterator over all values in this matrix,
     * proceeding through each row in order.
     */
    pub fn row_major_owned_iter(self) -> RowMajorOwnedIterator<T>
    where
        T: Default,
    {
        RowMajorOwnedIterator::new(self)
    }

    /**
     * Returns an iterator over references to the main diagonal in this matrix.
     */
    pub fn diagonal_reference_iter(&self) -> DiagonalReferenceIterator<T> {
        DiagonalReferenceIterator::new(self)
    }

    /**
     * Returns an iterator over mutable references to the main diagonal in this matrix.
     */
    pub fn diagonal_reference_mut_iter(&mut self) -> DiagonalReferenceMutIterator<T> {
        DiagonalReferenceMutIterator::new(self)
    }

    /**
     * Shrinks this matrix down from its current MxN size down to
     * some new size OxP where O and P are determined by the kind of
     * slice given and 1 <= O <= M and 1 <= P <= N.
     *
     * Only rows and columns specified by the slice will be retained, so for
     * instance if the Slice is constructed by
     * `Slice2D::new().rows(Slice::Range(0..2)).columns(Slice::Range(0..3))` then the
     * modified matrix will be no bigger than 2x3 and contain up to the first two
     * rows and first three columns that it previously had.
     *
     * See [Slice](slices::Slice) for constructing slices.
     *
     * # Panics
     *
     * This function will panic if the slice would delete all rows or all columns
     * from this matrix, ie the resulting matrix must be at least 1x1.
     */
    #[track_caller]
    pub fn retain_mut(&mut self, slice: Slice2D) {
        let mut r = 0;
        let mut c = 0;
        // drop the values rejected by the slice
        let columns = self.columns();
        self.data.retain(|_| {
            let keep = slice.accepts(r, c);
            if c < (columns - 1) {
                c += 1;
            } else {
                r += 1;
                c = 0;
            }
            keep
        });
        // work out the resulting size of this matrix by using the non
        // public fields of the Slice2D to handle each row and column
        // seperately.
        let remaining_rows = {
            let mut accepted = 0;
            for i in 0..self.rows() {
                if slice.rows.accepts(i) {
                    accepted += 1;
                }
            }
            accepted
        };
        let remaining_columns = {
            let mut accepted = 0;
            for i in 0..self.columns() {
                if slice.columns.accepts(i) {
                    accepted += 1;
                }
            }
            accepted
        };
        assert!(
            remaining_rows > 0,
            "Provided slice must leave at least 1 row in the retained matrix"
        );
        assert!(
            remaining_columns > 0,
            "Provided slice must leave at least 1 column in the retained matrix"
        );
        assert!(
            !self.data.is_empty(),
            "Provided slice must leave at least 1 row and 1 column in the retained matrix"
        );
        self.rows = remaining_rows;
        self.columns = remaining_columns
        // By construction jagged slices should be impossible, if this
        // invariant later changes by accident it would be possible to break the
        // rectangle shape invariant on a matrix object
        // As Slice2D should prevent the construction of jagged slices no
        // check is here to detect if all rows are still the same length
    }

    /**
     * Consumes a 1x1 matrix and converts it into a scalar without copying the data.
     *
     * # Example
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * # fn main() -> Result<(), Box<dyn std::error::Error>> {
     * let x = Matrix::column(vec![ 1.0, 2.0, 3.0 ]);
     * let sum_of_squares: f64 = (x.transpose() * x).try_into_scalar()?;
     * # Ok(())
     * # }
     * ```
     */
    pub fn try_into_scalar(self) -> Result<T, ScalarConversionError> {
        if self.size() == (1, 1) {
            Ok(self.data.into_iter().next().unwrap())
        } else {
            Err(ScalarConversionError {})
        }
    }

    /**
     * Partition a matrix into an arbitary number of non overlapping parts.
     *
     * **This function is much like a hammer you should be careful to not overuse. If you don't need
     * to mutate the parts of the matrix data individually it will be much easier and less error
     * prone to create immutable views into the matrix using [MatrixRange] instead.**
     *
     * Parts are returned in row major order, forming a grid of slices into the Matrix data that
     * can be mutated independently.
     *
     * # Panics
     *
     * Panics if any row or column index is greater than the number of rows or columns in the
     * matrix. Each list of row partitions and column partitions must also be in ascending order.
     *
     * # Further Info
     *
     * The partitions form the boundries between each slice of matrix data. Hence, for each
     * dimension, each partition may range between 0 and the length of the dimension inclusive.
     *
     * For one dimension of length 5, you can supply 0 up to 6 partitions,
     * `[0,1,2,3,4,5]` would split that dimension into 7, 0 to 0, 0 to 1, 1 to 2,
     * 2 to 3, 3 to 4, 4 to 5 and 5 to 5. 0 to 0 and 5 to 5 would of course be empty and the
     * 5 parts in between would each be of length 1 along that dimension.
     * `[2,4]` would instead split that dimension into three parts of 0 to 2, 2 to 4, and 4 to 5.
     * `[]` would not split that dimension at all, and give a single part of 0 to 5.
     *
     * `partition` does this along both dimensions, and returns the parts in row major order, so
     * you will receive a list of R+1 * C+1 length where R is the length of the row partitions
     * provided and C is the length of the column partitions provided. If you just want to split
     * a matrix into a 2x2 grid see [`partition_quadrants`](Matrix::partition_quadrants) which
     * provides a dedicated API with more ergonomics for extracting the parts.
     */
    #[track_caller]
    pub fn partition(
        &mut self,
        row_partitions: &[Row],
        column_partitions: &[Column],
    ) -> Vec<MatrixView<T, MatrixPart<T>>> {
        let rows = self.rows();
        let columns = self.columns();
        fn check_axis(partitions: &[usize], length: usize) {
            let mut previous: Option<usize> = None;
            for &index in partitions {
                assert!(index <= length);
                previous = match previous {
                    None => Some(index),
                    Some(i) => {
                        assert!(index > i, "{:?} must be ascending", partitions);
                        Some(i)
                    }
                }
            }
        }
        check_axis(row_partitions, rows);
        check_axis(column_partitions, columns);

        // There will be one more slice than partitions, since partitions are the boundries
        // between slices.
        let row_slices = row_partitions.len() + 1;
        let column_slices = column_partitions.len() + 1;
        let total_slices = row_slices * column_slices;
        let mut slices: Vec<Vec<&mut [T]>> = Vec::with_capacity(total_slices);
        let (_, mut data) = self.data.split_at_mut(0);

        let mut index = 0;
        for r in 0..row_slices {
            let row_index = row_partitions.get(r).cloned().unwrap_or(rows);
            // Determine how many rows of our matrix we need for the next set of row slices
            let rows_included = row_index - index;
            for _ in 0..column_slices {
                slices.push(Vec::with_capacity(rows_included));
            }
            index = row_index;

            for _ in 0..rows_included {
                // Partition the next row of our matrix along the columns
                let mut index = 0;
                for c in 0..column_slices {
                    let column_index = column_partitions.get(c).cloned().unwrap_or(columns);
                    let columns_included = column_index - index;
                    index = column_index;
                    // Split off as many elements as included in this column slice
                    let (slice, rest) = data.split_at_mut(columns_included);
                    // Insert the slice into the slices, we'll push `rows_included` times into
                    // each slice Vec.
                    slices[(r * column_slices) + c].push(slice);
                    data = rest;
                }
            }
        }
        // rest is now empty, so we can ignore it.

        slices
            .into_iter()
            .map(|slices| {
                let rows = slices.len();
                let columns = slices.first().map(|columns| columns.len()).unwrap_or(0);
                if columns == 0 {
                    // We may have allocated N rows but if each column in that row has no size
                    // our actual size is 0x0
                    MatrixView::from(MatrixPart::new(slices, 0, 0))
                } else {
                    MatrixView::from(MatrixPart::new(slices, rows, columns))
                }
            })
            .collect()
    }

    /**
     * Partition a matrix into 4 non overlapping quadrants. Top left starts at 0,0 until
     * exclusive of row and column, bottom right starts at row and column to the end of the matrix.
     *
     * # Panics
     *
     * Panics if the row or column are greater than the number of rows or columns in the matrix.
     *
     * # Examples
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * let mut matrix = Matrix::from(vec![
     *     vec![ 0, 1, 2 ],
     *     vec![ 3, 4, 5 ],
     *     vec![ 6, 7, 8 ]
     * ]);
     * // Split the matrix at the second row and first column giving 2x1, 2x2, 1x1 and 2x1
     * // quadrants.
     * // 0 | 1 2
     * // 3 | 4 5
     * // -------
     * // 6 | 7 8
     * let mut parts = matrix.partition_quadrants(2, 1);
     * assert_eq!(parts.top_left, Matrix::column(vec![ 0, 3 ]));
     * assert_eq!(parts.top_right, Matrix::from(vec![vec![ 1, 2 ], vec![ 4, 5 ]]));
     * assert_eq!(parts.bottom_left, Matrix::column(vec![ 6 ]));
     * assert_eq!(parts.bottom_right, Matrix::row(vec![ 7, 8 ]));
     * // Modify the matrix data independently without worrying about the borrow checker
     * parts.top_right.map_mut(|x| x + 10);
     * parts.bottom_left.map_mut(|x| x - 10);
     * // Drop MatrixQuadrants so we can use the matrix directly again
     * std::mem::drop(parts);
     * assert_eq!(matrix, Matrix::from(vec![
     *     vec![ 0, 11, 12 ],
     *     vec![ 3, 14, 15 ],
     *     vec![ -4, 7, 8 ]
     * ]));
     * ```
     */
    #[track_caller]
    #[allow(clippy::needless_lifetimes)] // false positive?
    pub fn partition_quadrants<'a>(
        &'a mut self,
        row: Row,
        column: Column,
    ) -> MatrixQuadrants<'a, T> {
        let mut parts = self.partition(&[row], &[column]).into_iter();
        // We know there will be exactly 4 parts returned by the partition since we provided
        // 1 row and 1 column to partition ourself into 4 with.
        MatrixQuadrants {
            top_left: parts.next().unwrap(),
            top_right: parts.next().unwrap(),
            bottom_left: parts.next().unwrap(),
            bottom_right: parts.next().unwrap(),
        }
    }

    /**
     * Returns a MatrixView giving a view of only the data within the row and column
     * [IndexRange]s.
     *
     * This is a shorthand for constructing the MatrixView from this Matrix.
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * use easy_ml::matrices::views::{MatrixView, MatrixRange, IndexRange};
     * let ab = Matrix::from(vec![
     *     vec![ 0, 1, 2, 0 ],
     *     vec![ 3, 4, 5, 1 ]
     * ]);
     * let shorter = ab.range(0..2, 1..3);
     * assert_eq!(
     *     shorter,
     *     Matrix::from(vec![
     *        vec![ 1, 2 ],
     *        vec![ 4, 5 ]
     *     ])
     * );
     * ```
     */
    pub fn range<R>(&self, rows: R, columns: R) -> MatrixView<T, MatrixRange<T, &Matrix<T>>>
    where
        R: Into<IndexRange>,
    {
        MatrixView::from(MatrixRange::from(self, rows, columns))
    }

    /**
     * Returns a MatrixView giving a view of only the data within the row and column
     * [IndexRange]s. The MatrixRange mutably borrows this Matrix, and can
     * therefore mutate it.
     *
     * This is a shorthand for constructing the MatrixView from this Matrix.
     */
    pub fn range_mut<R>(
        &mut self,
        rows: R,
        columns: R,
    ) -> MatrixView<T, MatrixRange<T, &mut Matrix<T>>>
    where
        R: Into<IndexRange>,
    {
        MatrixView::from(MatrixRange::from(self, rows, columns))
    }

    /**
     * Returns a MatrixView giving a view of only the data within the row and column
     * [IndexRange]s. The MatrixRange takes ownership of this Matrix, and can
     * therefore mutate it.
     *
     * This is a shorthand for constructing the MatrixView from this Matrix.
     */
    pub fn range_owned<R>(self, rows: R, columns: R) -> MatrixView<T, MatrixRange<T, Matrix<T>>>
    where
        R: Into<IndexRange>,
    {
        MatrixView::from(MatrixRange::from(self, rows, columns))
    }

    /**
     * Returns a MatrixView giving a view of only the data outside the row and column
     * [IndexRange]s.
     *
     * This is a shorthand for constructing the MatrixView from this Matrix.
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * use easy_ml::matrices::views::{MatrixView, MatrixMask, IndexRange};
     * let ab = Matrix::from(vec![
     *     vec![ 0, 1, 2, 0 ],
     *     vec![ 3, 4, 5, 1 ]
     * ]);
     * let shorter = ab.mask(0..1, 1..3);
     * assert_eq!(
     *     shorter,
     *     Matrix::from(vec![
     *        vec![ 3, 1 ]
     *     ])
     * );
     * ```
     */
    pub fn mask<R>(&self, rows: R, columns: R) -> MatrixView<T, MatrixMask<T, &Matrix<T>>>
    where
        R: Into<IndexRange>,
    {
        MatrixView::from(MatrixMask::from(self, rows, columns))
    }

    /**
     * Returns a MatrixView giving a view of only the data outside the row and column
     * [IndexRange]s. The MatrixMask mutably borrows this Matrix, and can
     * therefore mutate it.
     *
     * This is a shorthand for constructing the MatrixView from this Matrix.
     */
    pub fn mask_mut<R>(
        &mut self,
        rows: R,
        columns: R,
    ) -> MatrixView<T, MatrixMask<T, &mut Matrix<T>>>
    where
        R: Into<IndexRange>,
    {
        MatrixView::from(MatrixMask::from(self, rows, columns))
    }

    /**
     * Returns a MatrixView giving a view of only the data outside the row and column
     * [IndexRange]s. The MatrixMask takes ownership of this Matrix, and can
     * therefore mutate it.
     *
     * This is a shorthand for constructing the MatrixView from this Matrix.
     */
    pub fn mask_owned<R>(self, rows: R, columns: R) -> MatrixView<T, MatrixMask<T, Matrix<T>>>
    where
        R: Into<IndexRange>,
    {
        MatrixView::from(MatrixMask::from(self, rows, columns))
    }

    /**
     * Returns a MatrixView with the rows and columns specified reversed in iteration
     * order. The data of this matrix and the dimension lengths remain unchanged.
     *
     * This is a shorthand for constructing the MatrixView from this Matrix.
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
    pub fn reverse(&self, reverse: Reverse) -> MatrixView<T, MatrixReverse<T, &Matrix<T>>> {
        MatrixView::from(MatrixReverse::from(self, reverse))
    }

    /**
     * Returns a MatrixView with the rows and columns specified reversed in iteration
     * order. The data of this matrix and the dimension lengths remain unchanged. The MatrixReverse
     * mutably borrows this Matrix, and can therefore mutate it
     *
     * This is a shorthand for constructing the MatrixView from this Matrix.
     */
    pub fn reverse_mut(
        &mut self,
        reverse: Reverse,
    ) -> MatrixView<T, MatrixReverse<T, &mut Matrix<T>>> {
        MatrixView::from(MatrixReverse::from(self, reverse))
    }

    /**
     * Returns a MatrixView with the rows and columns specified reversed in iteration
     * order. The data of this matrix and the dimension lengths remain unchanged. The MatrixReverse
     * takes ownership of this Matrix, and can therefore mutate it
     *
     * This is a shorthand for constructing the MatrixView from this Matrix.
     */
    pub fn reverse_owned(self, reverse: Reverse) -> MatrixView<T, MatrixReverse<T, Matrix<T>>> {
        MatrixView::from(MatrixReverse::from(self, reverse))
    }

    /**
     * Converts this Matrix into a 2 dimensional Tensor with the provided dimension names.
     *
     * This is a wrapper around the `TryFrom<(Matrix<T>, [Dimension; 2])>` implementation.
     *
     * The Tensor will have the data in the same order, a shape with lengths of `self.rows()` then
     * `self.columns()` and the provided dimension names respectively.
     *
     * Result::Err is returned if the `rows` and `columns` dimension names are the same.
     */
    pub fn into_tensor(
        self,
        rows: crate::tensors::Dimension,
        columns: crate::tensors::Dimension,
    ) -> Result<crate::tensors::Tensor<T, 2>, crate::tensors::InvalidShapeError<2>> {
        (self, [rows, columns]).try_into()
    }
}

/**
 * Methods for matrices with types that can be copied, but still not neccessarily numerical.
 */
impl<T: Clone> Matrix<T> {
    /**
     * Computes and returns the transpose of this matrix
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * let x = Matrix::from(vec![
     *    vec![ 1, 2 ],
     *    vec![ 3, 4 ]]);
     * let y = Matrix::from(vec![
     *    vec![ 1, 3 ],
     *    vec![ 2, 4 ]]);
     * assert_eq!(x.transpose(), y);
     * ```
     */
    pub fn transpose(&self) -> Matrix<T> {
        Matrix::from_fn((self.columns(), self.rows()), |(column, row)| {
            self.get(row, column)
        })
    }

    /**
     * Transposes the matrix in place (if it is square).
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * let mut x = Matrix::from(vec![
     *    vec![ 1, 2 ],
     *    vec![ 3, 4 ]]);
     * x.transpose_mut();
     * let y = Matrix::from(vec![
     *    vec![ 1, 3 ],
     *    vec![ 2, 4 ]]);
     * assert_eq!(x, y);
     * ```
     *
     * Note: None square matrices were erroneously not supported in previous versions (<=1.8.0) and
     * could be incorrectly mutated. This method will now correctly transpose non square matrices
     * by not attempting to transpose them in place.
     */
    pub fn transpose_mut(&mut self) {
        if self.rows() != self.columns() {
            let transposed = self.transpose();
            self.data = transposed.data;
            self.rows = transposed.rows;
            self.columns = transposed.columns;
        } else {
            for i in 0..self.rows() {
                for j in 0..self.columns() {
                    if i > j {
                        continue;
                    }
                    let temp = self.get(i, j);
                    self.set(i, j, self.get(j, i));
                    self.set(j, i, temp);
                }
            }
        }
    }

    /**
     * Returns an iterator over a column vector in this matrix. Columns are 0 indexed.
     *
     * If you have a matrix such as:
     * ```ignore
     * [
     *    1, 2, 3
     *    4, 5, 6
     *    7, 8, 9
     * ]
     * ```
     * then a column of 0, 1, and 2 will yield [1, 4, 7], [2, 5, 8] and [3, 6, 9]
     * respectively. If you do not need to copy the elements use
     * [`column_reference_iter`](Matrix::column_reference_iter) instead.
     *
     * # Panics
     *
     * Panics if the column does not exist in this matrix.
     */
    #[track_caller]
    pub fn column_iter(&self, column: Column) -> ColumnIterator<T> {
        ColumnIterator::new(self, column)
    }

    /**
     * Returns an iterator over a row vector in this matrix. Rows are 0 indexed.
     *
     * If you have a matrix such as:
     * ```ignore
     * [
     *    1, 2, 3
     *    4, 5, 6
     *    7, 8, 9
     * ]
     * ```
     * then a row of 0, 1, and 2 will yield [1, 2, 3], [4, 5, 6] and [7, 8, 9]
     * respectively. If you do not need to copy the elements use
     * [`row_reference_iter`](Matrix::row_reference_iter) instead.
     *
     * # Panics
     *
     * Panics if the row does not exist in this matrix.
     */
    #[track_caller]
    pub fn row_iter(&self, row: Row) -> RowIterator<T> {
        RowIterator::new(self, row)
    }

    /**
     * Returns a column major iterator over all values in this matrix, proceeding through each
     * column in order.
     *
     * If you have a matrix such as:
     * ```ignore
     * [
     *    1, 2
     *    3, 4
     * ]
     * ```
     * then the iterator will yield [1, 3, 2, 4]. If you do not need to copy the
     * elements use [`column_major_reference_iter`](Matrix::column_major_reference_iter) instead.
     */
    pub fn column_major_iter(&self) -> ColumnMajorIterator<T> {
        ColumnMajorIterator::new(self)
    }

    /**
     * Returns a row major iterator over all values in this matrix, proceeding through each
     * row in order.
     *
     * If you have a matrix such as:
     * ```ignore
     * [
     *    1, 2
     *    3, 4
     * ]
     * ```
     * then the iterator will yield [1, 2, 3, 4]. If you do not need to copy the
     * elements use [`row_major_reference_iter`](Matrix::row_major_reference_iter) instead.
     */
    pub fn row_major_iter(&self) -> RowMajorIterator<T> {
        RowMajorIterator::new(self)
    }

    /**
     * Returns a iterator over the main diagonal of this matrix.
     *
     * If you have a matrix such as:
     * ```ignore
     * [
     *    1, 2
     *    3, 4
     * ]
     * ```
     * then the iterator will yield [1, 4]. If you do not need to copy the
     * elements use [`diagonal_reference_iter`](Matrix::diagonal_reference_iter) instead.
     *
     * # Examples
     *
     * Computing a [trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra))
     * ```
     * use easy_ml::matrices::Matrix;
     * let matrix = Matrix::from(vec![
     *     vec![ 1, 2, 3 ],
     *     vec![ 4, 5, 6 ],
     *     vec![ 7, 8, 9 ],
     * ]);
     * let trace: i32 = matrix.diagonal_iter().sum();
     * assert_eq!(trace, 1 + 5 + 9);
     * ```
     */
    pub fn diagonal_iter(&self) -> DiagonalIterator<T> {
        DiagonalIterator::new(self)
    }

    /**
     * Creates a matrix of the provided size with all elements initialised to the provided value
     *
     * # Panics
     *
     * Panics if no values are provided. Note: this method erroneously did not validate its inputs
     * in Easy ML versions up to and including 1.7.0
     */
    #[track_caller]
    pub fn empty(value: T, size: (Row, Column)) -> Matrix<T> {
        assert!(size.0 > 0 && size.1 > 0, "Size must be at least 1x1");
        Matrix {
            data: vec![value; size.0 * size.1],
            rows: size.0,
            columns: size.1,
        }
    }

    /**
     * Gets a copy of the value at this row and column. Rows and Columns are 0 indexed.
     *
     * # Panics
     *
     * Panics if the index is out of range.
     */
    #[track_caller]
    pub fn get(&self, row: Row, column: Column) -> T {
        assert!(
            row < self.rows(),
            "Row out of index, only have {} rows",
            self.rows()
        );
        assert!(
            column < self.columns(),
            "Column out of index, only have {} columns",
            self.columns()
        );
        self.data[self.get_index(row, column)].clone()
    }

    /**
     * Similar to matrix.get(0, 0) in that this returns the element in the first
     * row and first column, except that this method will panic if the matrix is
     * not 1x1.
     *
     * This is provided as a convenience function when you want to convert a unit matrix
     * to a scalar, such as after taking a dot product of two vectors.
     *
     * # Example
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * let x = Matrix::column(vec![ 1.0, 2.0, 3.0 ]);
     * let sum_of_squares: f64 = (x.transpose() * x).scalar();
     * ```
     *
     * # Panics
     *
     * Panics if the matrix is not 1x1
     */
    #[track_caller]
    pub fn scalar(&self) -> T {
        assert!(
            self.rows() == 1,
            "Cannot treat matrix as scalar as it has more than one row"
        );
        assert!(
            self.columns() == 1,
            "Cannot treat matrix as scalar as it has more than one column"
        );
        self.get(0, 0)
    }

    /**
     * Applies a function to all values in the matrix, modifying
     * the matrix in place.
     */
    pub fn map_mut(&mut self, mapping_function: impl Fn(T) -> T) {
        for value in self.data.iter_mut() {
            *value = mapping_function(value.clone());
        }
    }

    /**
     * Applies a function to all values and each value's index in the
     * matrix, modifying the matrix in place.
     */
    pub fn map_mut_with_index(&mut self, mapping_function: impl Fn(T, Row, Column) -> T) {
        self.row_major_reference_mut_iter()
            .with_index()
            .for_each(|((i, j), x)| {
                *x = mapping_function(x.clone(), i, j);
            });
    }

    /**
     * Creates and returns a new matrix with all values from the original with the
     * function applied to each. This can be used to change the type of the matrix
     * such as creating a mask:
     * ```
     * use easy_ml::matrices::Matrix;
     * let x = Matrix::from(vec![
     *    vec![ 0.0, 1.2 ],
     *    vec![ 5.8, 6.9 ]]);
     * let y = x.map(|element| element > 2.0);
     * let result = Matrix::from(vec![
     *    vec![ false, false ],
     *    vec![ true, true ]]);
     * assert_eq!(&y, &result);
     * ```
     */
    pub fn map<U>(&self, mapping_function: impl Fn(T) -> U) -> Matrix<U>
    where
        U: Clone,
    {
        let mapped = self
            .data
            .iter()
            .map(|x| mapping_function(x.clone()))
            .collect();
        Matrix::from_flat_row_major(self.size(), mapped)
    }

    /**
     * Creates and returns a new matrix with all values from the original
     * and the index of each value mapped by a function. This can be used
     * to perform elementwise operations that are not defined on the
     * Matrix type itself.
     *
     * # Exmples
     *
     * Matrix elementwise division:
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * let x = Matrix::from(vec![
     *     vec![ 9.0, 2.0 ],
     *     vec![ 4.0, 3.0 ]]);
     * let y = Matrix::from(vec![
     *     vec![ 3.0, 2.0 ],
     *     vec![ 1.0, 3.0 ]]);
     * let z = x.map_with_index(|x, row, column| x / y.get(row, column));
     * let result = Matrix::from(vec![
     *     vec![ 3.0, 1.0 ],
     *     vec![ 4.0, 1.0 ]]);
     * assert_eq!(&z, &result);
     * ```
     */
    pub fn map_with_index<U>(&self, mapping_function: impl Fn(T, Row, Column) -> U) -> Matrix<U>
    where
        U: Clone,
    {
        let mapped = self
            .row_major_iter()
            .with_index()
            .map(|((i, j), x)| mapping_function(x, i, j))
            .collect();
        Matrix::from_flat_row_major(self.size(), mapped)
    }

    /**
     * Inserts a new row into the Matrix at the provided index,
     * shifting other rows to the right and filling all entries with the
     * provided value. Rows are 0 indexed.
     *
     * # Panics
     *
     * This will panic if the row is greater than the number of rows in the matrix.
     */
    #[track_caller]
    pub fn insert_row(&mut self, row: Row, value: T) {
        assert!(
            row <= self.rows(),
            "Row to insert must be <= to {}",
            self.rows()
        );
        for column in 0..self.columns() {
            self.data.insert(self.get_index(row, column), value.clone());
        }
        self.rows += 1;
    }

    /**
     * Inserts a new row into the Matrix at the provided index, shifting other rows
     * to the right and filling all entries with the values from the iterator in sequence.
     * Rows are 0 indexed.
     *
     * # Panics
     *
     * This will panic if the row is greater than the number of rows in the matrix,
     * or if the iterator has fewer elements than `self.columns()`.
     *
     * Example of duplicating a row:
     * ```
     * use easy_ml::matrices::Matrix;
     * let x: Matrix<u8> = Matrix::row(vec![ 1, 2, 3 ]);
     * let mut y = x.clone();
     * // duplicate the first row as the second row
     * y.insert_row_with(1, x.row_iter(0));
     * assert_eq!((2, 3), y.size());
     * let mut values = y.column_major_iter();
     * assert_eq!(Some(1), values.next());
     * assert_eq!(Some(1), values.next());
     * assert_eq!(Some(2), values.next());
     * assert_eq!(Some(2), values.next());
     * assert_eq!(Some(3), values.next());
     * assert_eq!(Some(3), values.next());
     * assert_eq!(None, values.next());
     * ```
     */
    #[track_caller]
    pub fn insert_row_with<I>(&mut self, row: Row, mut values: I)
    where
        I: Iterator<Item = T>,
    {
        assert!(
            row <= self.rows(),
            "Row to insert must be <= to {}",
            self.rows()
        );
        for column in 0..self.columns() {
            self.data.insert(
                self.get_index(row, column),
                values.next().unwrap_or_else(|| {
                    panic!("At least {} values must be provided", self.columns())
                }),
            );
        }
        self.rows += 1;
    }

    /**
     * Inserts a new column into the Matrix at the provided index, shifting other
     * columns to the right and filling all entries with the provided value.
     * Columns are 0 indexed.
     *
     * # Panics
     *
     * This will panic if the column is greater than the number of columns in the matrix.
     */
    #[track_caller]
    pub fn insert_column(&mut self, column: Column, value: T) {
        assert!(
            column <= self.columns(),
            "Column to insert must be <= to {}",
            self.columns()
        );
        for row in (0..self.rows()).rev() {
            self.data.insert(self.get_index(row, column), value.clone());
        }
        self.columns += 1;
    }

    /**
     * Inserts a new column into the Matrix at the provided index, shifting other columns
     * to the right and filling all entries with the values from the iterator in sequence.
     * Columns are 0 indexed.
     *
     * # Panics
     *
     * This will panic if the column is greater than the number of columns in the matrix,
     * or if the iterator has fewer elements than `self.rows()`.
     *
     * Example of duplicating a column:
     * ```
     * use easy_ml::matrices::Matrix;
     * let x: Matrix<u8> = Matrix::column(vec![ 1, 2, 3 ]);
     * let mut y = x.clone();
     * // duplicate the first column as the second column
     * y.insert_column_with(1, x.column_iter(0));
     * assert_eq!((3, 2), y.size());
     * let mut values = y.column_major_iter();
     * assert_eq!(Some(1), values.next());
     * assert_eq!(Some(2), values.next());
     * assert_eq!(Some(3), values.next());
     * assert_eq!(Some(1), values.next());
     * assert_eq!(Some(2), values.next());
     * assert_eq!(Some(3), values.next());
     * assert_eq!(None, values.next());
     * ```
     */
    #[track_caller]
    pub fn insert_column_with<I>(&mut self, column: Column, values: I)
    where
        I: Iterator<Item = T>,
    {
        assert!(
            column <= self.columns(),
            "Column to insert must be <= to {}",
            self.columns()
        );
        let mut array_values = values.collect::<Vec<T>>();
        assert!(
            array_values.len() >= self.rows(),
            "At least {} values must be provided",
            self.rows()
        );
        for row in (0..self.rows()).rev() {
            self.data
                .insert(self.get_index(row, column), array_values.pop().unwrap());
        }
        self.columns += 1;
    }

    /**
     * Makes a copy of this matrix shrunk down in size according to the slice. See
     * [retain_mut](Matrix::retain_mut()).
     */
    pub fn retain(&self, slice: Slice2D) -> Matrix<T> {
        let mut retained = self.clone();
        retained.retain_mut(slice);
        retained
    }
}

/**
 * Any matrix of a Cloneable type implements Clone.
 */
impl<T: Clone> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        self.map(|element| element)
    }
}

/**
 * Any matrix of a Displayable type implements Display
 *
 * You can control the precision of the formatting using format arguments, i.e.
 * `format!("{:.3}", matrix)`
 */
impl<T: std::fmt::Display> std::fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::matrices::views::format_view(self, f)
    }
}

/**
 * Any matrix and two different dimension names can be converted to a 2 dimensional tensor with
 * the same number of rows and columns.
 *
 * Conversion will fail if the dimension names for `self.rows()` and `self.columns()` respectively
 * are the same.
 */
impl<T> TryFrom<(Matrix<T>, [crate::tensors::Dimension; 2])> for crate::tensors::Tensor<T, 2> {
    type Error = crate::tensors::InvalidShapeError<2>;

    fn try_from(value: (Matrix<T>, [crate::tensors::Dimension; 2])) -> Result<Self, Self::Error> {
        let (matrix, [row_name, column_name]) = value;
        let shape = [(row_name, matrix.rows), (column_name, matrix.columns)];
        let check = crate::tensors::InvalidShapeError::new(shape);
        if !check.is_valid() {
            return Err(check);
        }
        // Now we know the shape is valid, we can call the standard Tensor constructor knowing
        // it won't fail since our data length will match the size of our shape.
        Ok(crate::tensors::Tensor::from(shape, matrix.data))
    }
}

/**
 * Methods for matrices with numerical types, such as f32 or f64.
 *
 * Note that unsigned integers are not Numeric because they do not
 * implement [Neg](std::ops::Neg). You must first
 * wrap unsigned integers via [Wrapping](std::num::Wrapping) or [Saturating](std::num::Saturating).
 *
 * While these methods will all be defined on signed integer types as well, such as i16 or i32,
 * in many cases integers cannot be used sensibly in these computations. If you
 * have a matrix of type i8 for example, you should consider mapping it into a floating
 * type before doing heavy linear algebra maths on it.
 *
 * Determinants can be computed without loss of precision using sufficiently large signed
 * integers because the only operations performed on the elements are addition, subtraction
 * and mulitplication. However the inverse of a matrix such as
 *
 * ```ignore
 * [
 *   4, 7
 *   2, 8
 * ]
 * ```
 *
 * is
 *
 * ```ignore
 * [
 *   0.6, -0.7,
 *  -0.2, 0.4
 * ]
 * ```
 *
 * which requires a type that supports decimals to accurately represent.
 *
 * Mapping matrix type example:
 * ```
 * use easy_ml::matrices::Matrix;
 * use std::num::Wrapping;
 *
 * let matrix: Matrix<u8> = Matrix::from(vec![
 *     vec![ 2, 3 ],
 *     vec![ 6, 0 ]
 * ]);
 * // determinant is not defined on this matrix because u8 is not Numeric
 * // println!("{:?}", matrix.determinant()); // won't compile
 * // however Wrapping<u8> is numeric
 * let matrix = matrix.map(|element| Wrapping(element));
 * println!("{:?}", matrix.determinant()); // -> 238 (overflow)
 * println!("{:?}", matrix.map(|element| element.0 as i16).determinant()); // -> -18
 * println!("{:?}", matrix.map(|element| element.0 as f32).determinant()); // -> -18.0
 * ```
 */
impl<T: Numeric> Matrix<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    /**
     * Returns the determinant of this square matrix, or None if the matrix
     * does not have a determinant. See [`linear_algebra`](super::linear_algebra::determinant())
     */
    pub fn determinant(&self) -> Option<T> {
        linear_algebra::determinant::<T>(self)
    }

    /**
     * Computes the inverse of a matrix provided that it exists. To have an inverse a
     * matrix must be square (same number of rows and columns) and it must also have a
     * non zero determinant. See [`linear_algebra`](super::linear_algebra::inverse())
     */
    pub fn inverse(&self) -> Option<Matrix<T>> {
        linear_algebra::inverse::<T>(self)
    }

    /**
     * Computes the covariance matrix for this NxM feature matrix, in which
     * each N'th row has M features to find the covariance and variance of. See
     * [`linear_algebra`](super::linear_algebra::covariance_column_features())
     */
    pub fn covariance_column_features(&self) -> Matrix<T> {
        linear_algebra::covariance_column_features::<T>(self)
    }

    /**
     * Computes the covariance matrix for this NxM feature matrix, in which
     * each M'th column has N features to find the covariance and variance of. See
     * [`linear_algebra`](super::linear_algebra::covariance_row_features())
     */
    pub fn covariance_row_features(&self) -> Matrix<T> {
        linear_algebra::covariance_row_features::<T>(self)
    }
}

/**
 * Methods for matrices with numerical real valued types, such as f32 or f64.
 *
 * This excludes signed and unsigned integers as they do not support decimal
 * precision and hence can't be used for operations like square roots.
 *
 * Third party fixed precision and infinite precision decimal types should
 * be able to implement all of the methods for [Real] and then utilise these functions.
 */
impl<T: Real> Matrix<T>
where
    for<'a> &'a T: RealRef<T>,
{
    /**
     * Computes the [L2 norm](https://en.wikipedia.org/wiki/Euclidean_vector#Length)
     * of this row or column vector, also referred to as the length or magnitude,
     * and written as ||x||, or sometimes |x|.
     *
     * ||**a**|| = sqrt(a<sub>1</sub><sup>2</sup> + a<sub>2</sub><sup>2</sup> + a<sub>3</sub><sup>2</sup>...) = sqrt(**a**<sup>T</sup> * **a**)
     *
     * This is a shorthand for `(x.transpose() * x).scalar().sqrt()` for
     * column vectors and `(x * x.transpose()).scalar().sqrt()` for row vectors, ie
     * the square root of the dot product of a vector with itself.
     *
     * The euclidean length can be used to compute a
     * [unit vector](https://en.wikipedia.org/wiki/Unit_vector), that is, a
     * vector with length of 1. This should not be confused with a unit matrix,
     * which is another name for an identity matrix.
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * let a = Matrix::column(vec![ 1.0, 2.0, 3.0 ]);
     * let length = a.euclidean_length(); // (1^2 + 2^2 + 3^2)^0.5
     * let unit = a / length;
     * assert_eq!(unit.euclidean_length(), 1.0);
     * ```
     *
     * # Panics
     *
     * If the matrix is not a vector, ie if it has more than one row and more than one
     * column.
     */
    #[track_caller]
    pub fn euclidean_length(&self) -> T {
        if self.columns() == 1 {
            // column vector
            (self.transpose() * self).scalar().sqrt()
        } else if self.rows() == 1 {
            // row vector
            (self * self.transpose()).scalar().sqrt()
        } else {
            panic!(
                "Cannot compute unit vector of a non vector, rows: {}, columns: {}",
                self.rows(),
                self.columns()
            );
        }
    }
}

// FIXME: want this to be callable in the main numeric impl block
impl<T: Numeric> Matrix<T> {
    /**
     * Creates a diagonal matrix of the provided size with the diagonal elements
     * set to the provided value and all other elements in the matrix set to 0.
     * A diagonal matrix is always square.
     *
     * The size is still taken as a tuple to facilitate creating a diagonal matrix
     * from the dimensionality of an existing one. If the provided value is 1 then
     * this will create an identity matrix.
     *
     * A 3 x 3 identity matrix:
     * ```ignore
     * [
     *   1, 0, 0
     *   0, 1, 0
     *   0, 0, 1
     * ]
     * ```
     *
     * # Panics
     *
     * If the provided size is not square.
     */
    #[track_caller]
    pub fn diagonal(value: T, size: (Row, Column)) -> Matrix<T> {
        assert!(size.0 == size.1);
        let mut matrix = Matrix::empty(T::zero(), size);
        for i in 0..size.0 {
            matrix.set(i, i, value.clone());
        }
        matrix
    }

    /**
     * Creates a diagonal matrix with the elements along the diagonal set to the
     * provided values and all other elements in the matrix set to 0.
     * A diagonal matrix is always square.
     *
     * Examples
     *
     * ```
     * use easy_ml::matrices::Matrix;
     * let matrix = Matrix::from_diagonal(vec![ 1, 1, 1 ]);
     * assert_eq!(matrix.size(), (3, 3));
     * let copy = Matrix::from_diagonal(matrix.diagonal_iter().collect());
     * assert_eq!(matrix, copy);
     * assert_eq!(matrix, Matrix::from(vec![
     *     vec![ 1, 0, 0 ],
     *     vec![ 0, 1, 0 ],
     *     vec![ 0, 0, 1 ],
     * ]))
     * ```
     */
    pub fn from_diagonal(values: Vec<T>) -> Matrix<T> {
        let mut matrix = Matrix::empty(T::zero(), (values.len(), values.len()));
        for (i, element) in values.into_iter().enumerate() {
            matrix.set(i, i, element);
        }
        matrix
    }
}

/**
 * PartialEq is implemented as two matrices are equal if and only if all their elements
 * are equal and they have the same size.
 */
impl<T: PartialEq> PartialEq for Matrix<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.rows() != other.rows() {
            return false;
        }
        if self.columns() != other.columns() {
            return false;
        }
        // perform elementwise check, return true only if every element in
        // each matrix is the same
        self.data.iter().zip(other.data.iter()).all(|(x, y)| x == y)
    }
}

#[test]
fn test_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<Matrix<f64>>();
}

#[test]
fn test_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Matrix<f64>>();
}

#[cfg(feature = "serde")]
mod serde_impls {
    use crate::matrices::{Column, Matrix, Row};
    use serde::{Deserialize, Deserializer};

    #[derive(Deserialize)]
    #[serde(rename = "Matrix")]
    struct MatrixDeserialize<T> {
        data: Vec<T>,
        rows: Row,
        columns: Column,
    }

    impl<'de, T> Deserialize<'de> for Matrix<T>
    where
        T: Deserialize<'de>,
    {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            MatrixDeserialize::<T>::deserialize(deserializer).map(|d| {
                // Safety: Use the no copy constructor that performs validation to prevent invalid
                // serialized data being created as a Matrix, which would then break all the
                // code that's relying on these invariants.
                Matrix::from_flat_row_major((d.rows, d.columns), d.data)
            })
        }
    }
}

#[cfg(feature = "serde")]
#[test]
fn test_serialize() {
    fn assert_serialize<T: Serialize>() {}
    assert_serialize::<Matrix<f64>>();
}

#[cfg(feature = "serde")]
#[test]
fn test_deserialize() {
    use serde::Deserialize;
    fn assert_deserialize<'de, T: Deserialize<'de>>() {}
    assert_deserialize::<Matrix<f64>>();
}

#[cfg(feature = "serde")]
#[test]
fn test_serialization_deserialization_loop() {
    #[rustfmt::skip]
    let matrix = Matrix::from(vec![
        vec![1,  2,  3,  4],
        vec![5,  6,  7,  8],
        vec![9, 10, 11, 12],
    ]);
    let encoded = toml::to_string(&matrix).unwrap();
    assert_eq!(
        encoded,
        r#"data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
rows = 3
columns = 4
"#,
    );
    let parsed: Result<Matrix<i32>, _> = toml::from_str(&encoded);
    assert!(parsed.is_ok());
    assert_eq!(matrix, parsed.unwrap())
}

#[cfg(feature = "serde")]
#[test]
#[should_panic]
fn test_deserialization_validation() {
    let _result: Result<Matrix<i32>, _> = toml::from_str(
        r#"data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
rows = 3
columns = 3
"#,
    );
}

#[test]
fn test_indexing() {
    let a = Matrix::from(vec![vec![1, 2], vec![3, 4]]);
    assert_eq!(a.get_index(0, 1), 1);
    assert_eq!(a.get_row_column(1), (0, 1));
    assert_eq!(a.get(0, 1), 2);
    let b = Matrix::from(vec![vec![1, 2, 3], vec![5, 6, 7]]);
    assert_eq!(b.get_index(1, 2), 5);
    assert_eq!(b.get_row_column(5), (1, 2));
    assert_eq!(b.get(1, 2), 7);
    assert_eq!(
        Matrix::from(vec![vec![0, 0], vec![0, 0], vec![0, 0]])
            .map_with_index(|_, r, c| format!("{:?}x{:?}", r, c)),
        Matrix::from(vec![
            vec!["0x0", "0x1"],
            vec!["1x0", "1x1"],
            vec!["2x0", "2x1"]
        ])
        .map(|x| x.to_owned())
    );
}
