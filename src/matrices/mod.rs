/*!
 * Generic matrix type.
 *
 * Matrices are generic over some type `T`. If `T` is [Numeric](super::numeric) then
 * the matrix can be used in a mathematical way.
 */

use std::ops::{Add, Div, Mul, Neg, Sub};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod errors;
pub mod iterators;
pub mod slices;
pub mod views;

pub use errors::ScalarConversionError;

use crate::linear_algebra;
use crate::matrices::iterators::{
    ColumnIterator, ColumnMajorIterator, ColumnMajorReferenceIterator, ColumnReferenceIterator,
    DiagonalIterator, DiagonalReferenceIterator, RowIterator, RowMajorIterator,
    RowMajorReferenceIterator, RowReferenceIterator,
};
use crate::matrices::slices::Slice2D;
use crate::numeric::extra::{Real, RealRef};
use crate::numeric::{Numeric, NumericRef};

/**
 * A general purpose matrix of some type. This type may implement
 * no traits, in which case the matrix will be rather useless. If the
 * type implements [`Clone`](std::clone::Clone)
 * most storage and accessor methods are defined and if the type implements
 * [`Numeric`](super::numeric) then the matrix can be used in
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
 * is dependent on the platform's `std::usize::MAX` value. Matrices with dimensions NxM
 * such that N * M < `std::usize::MAX` should not cause any errors in this library, but
 * attempting to expand their size further may cause panics and or errors. At the time of
 * writing it is no longer possible to construct or use matrices where the product of their
 * number of rows and columns exceed `std::usize::MAX`, but some constructor methods may be used
 * to attempt this. Concerned readers should note that on a 64 bit computer this maximum
 * value is 18,446,744,073,709,551,615 so running out of memory is likely to occur first.
 *
 * # Matrix layout and iterator performance
 *
 * [See iterators submodule for Matrix layout and iterator performance](iterators#matrix-layout-and-iterator-performance)
 */
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
impl <T> Matrix<T> {
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
     */
    pub fn row(values: Vec<T>) -> Matrix<T> {
        Matrix {
            columns: values.len(),
            data: values,
            rows: 1,
        }
    }

    /**
     * Creates a column vector (Nx1) from a list
     */
    pub fn column(values: Vec<T>) -> Matrix<T> {
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
     *     vec![ 8, 9, 3 ]]);
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
        assert!(values.iter().map(|x| x.len()).all(|x| x == values[0].len()), "Inconsistent size");
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
     *     8, 9, 3]);
     * ```
     *
     * This method is more efficient than [`Matrix::from`](Matrix::from())
     * but requires specifying the size explicitly and manually keeping track of where rows
     * start and stop.
     *
     * # Panics
     *
     * Panics if the length of the vec does not match the size of the matrix.
     */
    #[track_caller]
    pub fn from_flat_row_major(size: (Row, Column), values: Vec<T>) -> Matrix<T> {
        assert!(size.0 * size.1 == values.len(),
            "Inconsistent size, attempted to construct a {}x{} matrix but provided with {} elements.",
            size.0, size.1, values.len());
        Matrix {
            data: values,
            rows: size.0,
            columns: size.1,
        }
    }

    #[deprecated(since="1.1.0", note="Incorrect use of terminology, a unit matrix is another term for an identity matrix, please use `from_scalar` instead")]
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
     * Not public API because don't want to name clash with the method on MatrixRef
     * that calls this.
     */
    pub(crate) fn _try_get_reference(&self, row: Row, column: Column) -> Option<&T> {
        if row < self.rows() || column < self.columns() {
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
        self.data.get_unchecked(self.get_index(row, column))
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
        if row < self.rows() || column < self.columns() {
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
        let index = self.get_index(row, column);
        // borrow for get_index ends
        self.data.get_unchecked_mut(index)
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

    /**
     * Returns an iterator over references to the main diagonal in this matrix.
     */
    pub fn diagonal_reference_iter(&self) -> DiagonalReferenceIterator<T> {
        DiagonalReferenceIterator::new(self)
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
            "Provided slice must leave at least 1 row in the retained matrix");
        assert!(
            remaining_columns > 0,
            "Provided slice must leave at least 1 column in the retained matrix");
        assert!(
            !self.data.is_empty(),
            "Provided slice must leave at least 1 row and 1 column in the retained matrix");
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
        if self.size() == (1,1) {
            Ok(self.data.into_iter().next().unwrap())
        } else {
            Err(ScalarConversionError {})
        }
    }
}

/**
 * Methods for matrices with types that can be copied, but still not neccessarily numerical.
 */
impl <T: Clone> Matrix<T> {
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
        let mut result = Matrix::empty(self.get(0, 0), (self.columns(), self.rows()));
        for i in 0..self.columns() {
            for j in 0..self.rows() {
                result.set(i, j, self.get(j, i).clone());
            }
        }
        result
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
     * Note: None square matrices were erroneously not supported in previous versions (1.7.0) and
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
     */
    pub fn empty(value: T, size: (Row, Column)) -> Matrix<T> {
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
        assert!(row < self.rows(),
            "Row out of index, only have {} rows", self.rows());
        assert!(column < self.columns(),
            "Column out of index, only have {} columns", self.columns());
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
        assert!(self.rows() == 1, "Cannot treat matrix as scalar as it has more than one row");
        assert!(self.columns() == 1, "Cannot treat matrix as scalar as it has more than one column");
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
        for i in 0..self.rows() {
            for j in 0..self.columns() {
                self.set(i, j, mapping_function(self.get(i, j), i, j));
            }
        }
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
            where U: Clone {
        let mapped = self.data.iter().map(|x| mapping_function(x.clone())).collect();
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
            where U: Clone {
        // compute the first mapped value so we have a value of type U
        // to initialise the mapped matrix with
        let first_value: U = mapping_function(self.get(0, 0), 0, 0);
        let mut mapped = Matrix::empty(first_value, self.size());
        for i in 0..self.rows() {
            for j in 0..self.columns() {
                mapped.set(i, j, mapping_function(self.get(i, j), i, j));
            }
        }
        mapped
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
        assert!(row <= self.rows(), "Row to insert must be <= to {}", self.rows());
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
    where I: Iterator<Item = T> {
        assert!(row <= self.rows(), "Row to insert must be <= to {}", self.rows());
        for column in 0..self.columns() {
            self.data.insert(
                self.get_index(row, column),
                values.next()
                    .unwrap_or_else(|| panic!(
                        "At least {} values must be provided",
                        self.columns()
                    ))
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
        assert!(column <= self.columns(), "Column to insert must be <= to {}", self.columns());
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
    where I: Iterator<Item = T> {
        assert!(column <= self.columns(), "Column to insert must be <= to {}", self.columns());
        let mut array_values = values.collect::<Vec<T>>();
        assert!(array_values.len() >= self.rows(),
            "At least {} values must be provided", self.rows());
        for row in (0..self.rows()).rev() {
            self.data.insert(
                self.get_index(row, column),
                array_values.pop().unwrap());
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
impl <T: Clone> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        self.map(|element| element)
    }
}

/**
 * Any matrix of a Displayable type implements Display
 */
impl <T: std::fmt::Display> std::fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        crate::matrices::views::format_view(self, f)
    }
}

/**
 * Methods for matrices with numerical types, such as f32 or f64.
 *
 * Note that unsigned integers are not Numeric because they do not
 * implement [Neg](std::ops::Neg). You must first
 * wrap unsigned integers via [Wrapping](std::num::Wrapping).
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
impl <T: Numeric> Matrix<T>
where for<'a> &'a T: NumericRef<T> {
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
    pub fn inverse(&self) -> Option<Matrix<T>>
    where T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Div<Output = T> {
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
 * be able to implement all of the methods for [Real](super::numeric::extra::Real)
 * and then utilise these functions.
 */
impl <T: Numeric + Real> Matrix<T>
where for<'a> &'a T: NumericRef<T> + RealRef<T> {
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
            panic!("Cannot compute unit vector of a non vector, rows: {}, columns: {}",
                self.rows(), self.columns());
        }
    }
}

// FIXME: want this to be callable in the main numeric impl block
impl <T: Numeric> Matrix<T> {
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
impl <T: PartialEq> PartialEq for Matrix<T> {
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
        self.data.iter()
            .zip(other.data.iter())
            .all(|(x, y)| x == y)
    }
}

/**
 * Matrix multiplication for two referenced matrices.
 *
 * This is matrix multiplication such that a matrix of dimensionality (LxM) multiplied with
 * a matrix of dimensionality (MxN) yields a new matrix of dimensionality (LxN) with each element
 * corresponding to the sum of products of the ith row in the first matrix and the jth column in
 * the second matrix.
 *
 * Matrices of the wrong sizes will result in a panic. No broadcasting is performed, ie you cannot
 * multiply a (NxM) matrix by a (Nx1) column vector, you must transpose one of the arguments so
 * that the operation is valid.
 */
impl <T: Numeric> Mul for &Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    // Tell the compiler our output type is another matrix of type T
    type Output = Matrix<T>;

    #[track_caller]
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        // LxM * MxN -> LxN
        assert!(self.columns() == rhs.rows(),
            "Mismatched Matrices, left is {}x{}, right is {}x{}, * is only defined for MxN * NxL",
            self.rows(), self.columns(), rhs.rows(), rhs.columns());

        let mut result = Matrix::empty(self.get(0, 0), (self.rows(), rhs.columns()));
        for i in 0..self.rows() {
            for j in 0..rhs.columns() {
                // compute dot product for each element in the new matrix
                result.set(i, j,
                    self.row_reference_iter(i)
                    .zip(rhs.column_reference_iter(j))
                    .map(|(x, y)| x * y)
                    .sum());
            }
        }
        result
    }
}

/**
 * Matrix multiplication for two matrices.
 */
impl <T: Numeric> Mul for Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Matrix<T>;
    #[track_caller]
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

/**
 * Matrix multiplication for two matrices with one referenced.
 */
impl <T: Numeric> Mul<&Matrix<T>> for Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Matrix<T>;
    #[track_caller]
    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

/**
 * Matrix multiplication for two matrices with one referenced.
 */
impl <T: Numeric> Mul<Matrix<T>> for &Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Matrix<T>;
    #[track_caller]
    #[inline]
    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        self * &rhs
    }
}

/**
 * Elementwise addition for two referenced matrices.
 */
impl <T: Numeric> Add for &Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    // Tell the compiler our output type is another matrix of type T
    type Output = Matrix<T>;

    #[track_caller]
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        // LxM + LxM -> LxM
        assert!(self.size() == rhs.size(),
            "Mismatched Matrices, left is {}x{}, right is {}x{}, + is only defined for MxN + MxN",
            self.rows(), self.columns(), rhs.rows(), rhs.columns());

        let values = self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(x, y)| x + y)
            .collect();
        Matrix::from_flat_row_major(self.size(), values)
    }
}

/**
 * Elementwise addition for two matrices.
 */
impl <T: Numeric> Add for Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Matrix<T>;
    #[track_caller]
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

/**
 * Elementwise addition for two matrices with one referenced.
 */
impl <T: Numeric> Add<&Matrix<T>> for Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Matrix<T>;
    #[track_caller]
    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

/**
 * Elementwise addition for two matrices with one referenced.
 */
impl <T: Numeric> Add<Matrix<T>> for &Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Matrix<T>;
    #[track_caller]
    #[inline]
    fn add(self, rhs: Matrix<T>) -> Self::Output {
        self + &rhs
    }
}

/**
 * Elementwise subtraction for two referenced matrices.
 */
impl <T: Numeric> Sub for &Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    // Tell the compiler our output type is another matrix of type T
    type Output = Matrix<T>;

    #[track_caller]
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        // LxM - LxM -> LxM
        assert!(self.size() == rhs.size(),
            "Mismatched Matrices, left is {}x{}, right is {}x{}, - is only defined for MxN - MxN",
            self.rows(), self.columns(), rhs.rows(), rhs.columns());

        let values = self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(x, y)| x - y)
            .collect();
        Matrix::from_flat_row_major(self.size(), values)
    }
}

/**
 * Elementwise subtraction for two matrices.
 */
impl <T: Numeric> Sub for Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Matrix<T>;
    #[track_caller]
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

/**
 * Elementwise subtraction for two matrices with one referenced.
 */
impl <T: Numeric> Sub<&Matrix<T>> for Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Matrix<T>;
    #[track_caller]
    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}

/**
 * Elementwise subtraction for two matrices with one referenced.
 */
impl <T: Numeric> Sub<Matrix<T>> for &Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Matrix<T>;
    #[track_caller]
    #[inline]
    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        self - &rhs
    }
}

/**
 * Elementwise negation for a referenced matrix.
 */
impl <T: Numeric> Neg for &Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    // Tell the compiler our output type is another matrix of type T
    type Output = Matrix<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        self.map(|v| -v)
    }
}

/**
 * Elementwise negation for a matrix.
 */
impl <T: Numeric> Neg for Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    // Tell the compiler our output type is another matrix of type T
    type Output = Matrix<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        - &self
    }
}

macro_rules! matrix_scalar_reference_reference {
    (impl $op:tt for Matrix { fn $method:ident }) => {
        /**
         * Operation for a matrix and scalar by reference. The scalar is applied to
         * all elements, this is a shorthand for map().
         */
         impl <T: Numeric> $op<&T> for &Matrix<T>
         where for<'a> &'a T: NumericRef<T> {
             type Output = Matrix<T>;
             #[inline]
             fn $method(self, rhs: &T) -> Self::Output {
                 self.map(|x| (x).$method(rhs.clone()))
             }
         }
    }
}

macro_rules! matrix_scalar_value_value {
    (impl $op:tt for Matrix { fn $method:ident }) => {
        /**
         * Operation for a matrix and scalar by value. The scalar is applied to
         * all elements, this is a shorthand for map().
         */
         impl <T: Numeric> $op<T> for Matrix<T>
         where for<'a> &'a T: NumericRef<T> {
             type Output = Matrix<T>;
             #[inline]
             fn $method(self, rhs: T) -> Self::Output {
                 self.map(|x| (x).$method(rhs.clone()))
             }
         }
    }
}

macro_rules! matrix_scalar_value_reference {
    (impl $op:tt for Matrix { fn $method:ident }) => {
        /**
         * Operation for a matrix by value and scalar by reference. The scalar is applied to
         * all elements, this is a shorthand for map().
         */
         impl <T: Numeric> $op<&T> for Matrix<T>
         where for<'a> &'a T: NumericRef<T> {
             type Output = Matrix<T>;
             #[inline]
             fn $method(self, rhs: &T) -> Self::Output {
                 self.map(|x| (x).$method(rhs.clone()))
             }
         }
    }
}

macro_rules! matrix_scalar_reference_value {
    (impl $op:tt for Matrix { fn $method:ident }) => {
        /**
         * Operation for a matrix by reference and scalar by value. The scalar is applied to
         * all elements, this is a shorthand for map().
         */
         impl <T: Numeric> $op<T> for &Matrix<T>
         where for<'a> &'a T: NumericRef<T> {
             type Output = Matrix<T>;
             #[inline]
             fn $method(self, rhs: T) -> Self::Output {
                 self.map(|x| (x).$method(rhs.clone()))
             }
         }
    }
}

matrix_scalar_reference_reference!(impl Add for Matrix { fn add });
matrix_scalar_value_reference!(impl Add for Matrix { fn add });
matrix_scalar_reference_value!(impl Add for Matrix { fn add });
matrix_scalar_value_value!(impl Add for Matrix { fn add });

matrix_scalar_reference_reference!(impl Sub for Matrix { fn sub });
matrix_scalar_value_reference!(impl Sub for Matrix { fn sub });
matrix_scalar_reference_value!(impl Sub for Matrix { fn sub });
matrix_scalar_value_value!(impl Sub for Matrix { fn sub });

matrix_scalar_reference_reference!(impl Mul for Matrix { fn mul });
matrix_scalar_value_reference!(impl Mul for Matrix { fn mul });
matrix_scalar_reference_value!(impl Mul for Matrix { fn mul });
matrix_scalar_value_value!(impl Mul for Matrix { fn mul });

matrix_scalar_reference_reference!(impl Div for Matrix { fn div });
matrix_scalar_value_reference!(impl Div for Matrix { fn div });
matrix_scalar_reference_value!(impl Div for Matrix { fn div });
matrix_scalar_value_value!(impl Div for Matrix { fn div });

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
#[test]
fn test_serialize() {
    fn assert_serialize<T: Serialize>() {}
    assert_serialize::<Matrix<f64>>();
}

#[cfg(feature = "serde")]
#[test]
fn test_deserialize() {
    fn assert_deserialize<'de, T: Deserialize<'de>>() {}
    assert_deserialize::<Matrix<f64>>();
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
        Matrix::from(vec![
            vec![0, 0],
            vec![0, 0],
            vec![0, 0]
        ])
        .map_with_index(|_, r, c| format!("{:?}x{:?}", r, c)),
        Matrix::from(vec![
            vec!["0x0", "0x1"],
            vec!["1x0", "1x1"],
            vec!["2x0", "2x1"]
        ])
        .map(|x| x.to_owned())
    );
}
