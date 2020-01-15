/*!
 * Generic matrix type.
 *
 * Matrices are generic over some type `T`. If `T` is [Numeric](../numeric/index.html) then
 * the matrix can be used in a mathematical way.
 */

use std::ops::{Add, Sub, Mul, Neg, Div};

pub mod iterators;
pub mod slices;

use crate::matrices::iterators::{
    ColumnIterator, RowIterator, ColumnMajorIterator,
    ColumnReferenceIterator, RowReferenceIterator, ColumnMajorReferenceIterator};
use crate::matrices::slices::Slice;
use crate::numeric::{Numeric, NumericRef};
use crate::linear_algebra;

/**
 * A general purpose matrix of some type. This type may implement
 * no traits, in which case the matrix will be rather useless. If the
 * type implements [`Clone`](https://doc.rust-lang.org/std/clone/trait.Clone.html)
 * most storage and accessor methods are defined and if the type implements
 * [`Numeric`](../numeric/index.html) then the matrix can be used in
 * a mathematical way.
 *
 * When doing numeric operations with Matrices you should be careful to not
 * consume a matrix by accidentally using it by value. All the operations are
 * also defined on references to matrices so you should favor `&x * &y` style
 * notation for matrices you intend to continue using.
 */
#[derive(Debug)]
pub struct Matrix<T> {
    data: Vec<Vec<T>>
}

/// The maximum row and column lengths are usize, due to the internal storage being backed by
/// nested Vecs
pub type Row = usize;
pub type Column = usize;

/**
 * Methods for matrices of any type, including non numerical types such as bool.
 */
impl <T> Matrix<T> {
    /**
     * Creates a unit (1x1) matrix from some element
     */
    pub fn unit(value: T) -> Matrix<T> {
        Matrix {
            data: vec![vec![value]]
        }
    }

    /**
     * Creates a row vector (1xN) from a list
     */
    pub fn row(values: Vec<T>) -> Matrix<T> {
        Matrix {
            data: vec![values]
        }
    }

    /**
     * Creates a column vector (Nx1) from a list
     */
    pub fn column(values: Vec<T>) -> Matrix<T> {
        Matrix {
            data: values.into_iter().map(|x| vec![x]).collect()
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
     */
    pub fn from(values: Vec<Vec<T>>) -> Matrix<T> {
        assert!(!values.is_empty(), "No rows defined");
        // check length of first row is > 1
        assert!(!values[0].is_empty(), "No column defined");
        // check length of each row is the same
        assert!(values.iter().map(|x| x.len()).all(|x| x == values[0].len()), "Inconsistent size");
        Matrix {
            data: values
        }
    }

    /**
     * Returns the dimensionality of this matrix in Row, Column format
     */
    pub fn size(&self) -> (Row, Column) {
        (self.data.len(), self.data[0].len())
    }

    /**
     * Gets the number of rows in this matrix.
     */
    pub fn rows(&self) -> Row {
        self.data.len()
    }

    /**
     * Gets the number of columns in this matrix.
     */
    pub fn columns(&self) -> Column {
        self.data[0].len()
    }

    /**
     * Gets a reference to the value at this row and column. Rows and Columns are 0 indexed.
     */
    pub fn get_reference(&self, row: Row, column: Column) -> &T {
        assert!(row < self.rows(), "Row out of index");
        assert!(column < self.columns(), "Column out of index");
        &self.data[row][column]
    }

    /**
     * Sets a new value to this row and column. Rows and Columns are 0 indexed.
     */
    pub fn set(&mut self, row: Row, column: Column, value: T) {
        assert!(row < self.rows(), "Row out of index");
        assert!(column < self.columns(), "Column out of index");
        self.data[row][column] = value;
    }

    /**
     * Removes a row from this Matrix, shifting all other rows to the left.
     * Rows are 0 indexed.
     *
     * This will panic if the row does not exist or the matrix only has
     * one row.
     */
    pub fn remove_row(&mut self, row: Row) {
        assert!(self.rows() > 1);
        self.data.remove(row);
    }

    /**
     * Removes a column from this Matrix, shifting all other columns to the left.
     * Columns are 0 indexed.
     *
     * This will panic if the column does not exist or the matrix only has
     * one column.
     */
    pub fn remove_column(&mut self, column: Column) {
        assert!(self.columns() > 1);
        for row in 0..self.rows() {
            self.data[row].remove(column);
        }
    }

    /**
     * Returns an iterator over references to a column vector in this matrix.
     * Columns are 0 indexed.
     */
    pub fn column_reference_iter(&self, column: Column) -> ColumnReferenceIterator<T> {
        ColumnReferenceIterator::new(self, column)
    }

    /**
     * Returns an iterator over references to a row vector in this matrix.
     * Rows are 0 indexed.
     */
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
     * Shrinks this matrix down from its current MxN size down to
     * some new size OxP where O and P are determined by the kind of
     * slice given, O <= M and P <= N.
     *
     * Only rows and columns specified by the slice will be retained, so for
     * instance if the Slice is `Slice::RowColumnRange(0..2, 0..3)` then the
     * modified matrix will be no bigger than 2x3 and contain up to the first two
     * rows and first three columns that it previously had.
     *
     * See [Slice](./slices/enum.Slice.html) for constructing slices.
     */
    pub fn retain_mut(&mut self, slice: Slice) {
        // iterate through rows and columns backwards so removing entries doesn't
        // invalidate the index
        for row in (0..self.rows()).rev() {
            for column in (0..self.columns()).rev() {
                if !slice.accepts(row, column) {
                    self.data[row].remove(column);
                }
            }
            // delete empty rows
            if self.data[row].len() == 0 {
                self.data.remove(row);
            }
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
     * Transposes the matrix in place.
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
     */
    pub fn transpose_mut(&mut self) {
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
     * respectively. If you do not need to copy the elements use `column_reference_iter`
     * instead.
     */
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
     * respectively. If you do not need to copy the elements use `row_reference_iter`
     * instead.
     */
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
     * elements use `column_major_reference_iter` instead.
     */
    pub fn column_major_iter(&self) -> ColumnMajorIterator<T> {
        ColumnMajorIterator::new(self)
    }

    /**
     * Creates a matrix of the provided size with all elements initialised to the provided value
     */
    pub fn empty(value: T, size: (Row, Column)) -> Matrix<T> {
        Matrix {
            data: vec![vec![value; size.1]; size.0]
        }
    }

    /**
     * Gets a copy of the value at this row and column. Rows and Columns are 0 indexed.
     */
    pub fn get(&self, row: Row, column: Column) -> T {
        assert!(row < self.rows(), "Row out of index");
        assert!(column < self.columns(), "Column out of index");
        self.data[row][column].clone()
    }

    /**
     * Applies a function to all values in the matrix, modifying
     * the matrix in place.
     */
    pub fn map_mut(&mut self, mapping_function: impl Fn(T) -> T) {
        for i in 0..self.rows() {
            for j in 0..self.columns() {
                self.set(i, j, mapping_function(self.get(i, j).clone()));
            }
        }
    }

    /**
     * Applies a function to all values and each value's index in the
     * matrix, modifying the matrix in place.
     */
    pub fn map_mut_with_index(&mut self, mapping_function: impl Fn(T, Row, Column) -> T) {
        for i in 0..self.rows() {
            for j in 0..self.columns() {
                self.set(i, j, mapping_function(self.get(i, j).clone(), i, j));
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
        // compute the first mapped value so we have a value of type U
        // to initialise the mapped matrix with
        let first_value: U = mapping_function(self.get(0, 0));
        let mut mapped = Matrix::empty(first_value, self.size());
        for i in 0..self.rows() {
            for j in 0..self.columns() {
                mapped.set(i, j, mapping_function(self.get(i, j).clone()));
            }
        }
        mapped
    }

    /**
     * Creates and returns a new matrix with all values from the original
     * and the index of each value mapped by a function.
     */
    pub fn map_with_index<U>(&self, mapping_function: impl Fn(T, Row, Column) -> U) -> Matrix<U>
            where U: Clone {
        // compute the first mapped value so we have a value of type U
        // to initialise the mapped matrix with
        let first_value: U = mapping_function(self.get(0, 0), 0, 0);
        let mut mapped = Matrix::empty(first_value, self.size());
        for i in 0..self.rows() {
            for j in 0..self.columns() {
                mapped.set(i, j, mapping_function(self.get(i, j).clone(), i, j));
            }
        }
        mapped
    }

    /**
     * Inserts a new row into the Matrix at the provided index,
     * shifting other rows to the right and filling all entries with the
     * provided value. Rows are 0 indexed.
     *
     * This will panic if the row is greater than the number of rows in the matrix.
     */
    pub fn insert_row(&mut self, row: Row, value: T) {
        let new_row = vec![value; self.columns()];
        self.data.insert(row, new_row);
    }

    /**
     * Inserts a new row into the Matrix at the provided index, shifting other rows
     * to the right and filling all entries with the values from the iterator in sequence.
     * Rows are 0 indexed.
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
    pub fn insert_row_with<I>(&mut self, row: Row, values: I)
    where I: Iterator<Item = T> {
        let new_row = values.take(self.columns()).collect();
        self.data.insert(row, new_row);
    }

    /**
     * Inserts a new column into the Matrix at the provided index, shifting other
     * columns to the right and filling all entries with the provided value.
     * Columns are 0 indexed.
     *
     * This will panic if the column is greater than the number of columns in the matrix.
     */
    pub fn insert_column(&mut self, column: Column, value: T) {
        for row in 0..self.rows() {
            self.data[row].insert(column, value.clone());
        }
    }

    /**
     * Inserts a new column into the Matrix at the provided index, shifting other columns
     * to the right and filling all entries with the values from the iterator in sequence.
     * Columns are 0 indexed.
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
    pub fn insert_column_with<I>(&mut self, column: Column, mut values: I)
    where I: Iterator<Item = T> {
        for row in 0..self.rows() {
            self.data[row].insert(column, values.next().unwrap());
        }
    }

    /**
     * Makes a copy of this matrix shrunk down in size according to the slice. See
     * [retain_mut](#method.retain_mut).
     */
    pub fn retain(&self, slice: Slice) -> Matrix<T> {
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
 * Methods for matrices with numerical types, such as f32 or f64.
 *
 * Note that unsigned integers are not Numeric because they do not
 * implement [Neg](https://doc.rust-lang.org/std/ops/trait.Neg.html). You must first
 * wrap unsigned integers via [Wrapping](https://doc.rust-lang.org/std/num/struct.Wrapping.html).
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
     * does not have a determinant. See [`linear_algebra`](../linear_algebra/fn.determinant.html)
     */
    pub fn determinant(&self) -> Option<T> {
        linear_algebra::determinant(self)
    }

    /**
    * Computes the inverse of a matrix provided that it exists. To have an inverse a
    * matrix must be square (same number of rows and columns) and it must also have a
    * non zero determinant. See [`linear_algebra`](../linear_algebra/fn.inverse.html)
    */
    pub fn inverse(&self) -> Option<Matrix<T>>
    where T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Div<Output = T> {
        linear_algebra::inverse(self)
    }

    /**
     * Computes the covariance matrix for this NxM feature matrix, in which
     * each N'th row has M features to find the covariance and variance of.
     * Returns `None` if this type's maximum value is less than N. See
     * [`linear_algebra`](../linear_algebra/fn.covariance.html)
     */
    pub fn covariance(&self) -> Option<Matrix<T>> {
        linear_algebra::covariance(self)
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
     */
    pub fn diagonal(value: T, size: (Row, Column)) -> Matrix<T> {
        assert!(size.0 == size.1);
        let mut matrix = Matrix {
            data: vec![vec![T::zero(); size.1]; size.0]
        };
        for i in 0..size.0 {
            matrix.set(i, i, value.clone());
        }
        matrix
    }
}

/**
 * PartialEq is implemented as two matrices are equal if and only if all their elements
 * are equal and they have the same size.
 */
impl <T: PartialEq> PartialEq for Matrix<T> {
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
            .all(|(x, y)| x.iter().zip(y.iter()).all(|(a, b)| a == b))
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

    fn mul(self, rhs: Self) -> Self::Output {
        // LxM * MxN -> LxN
        assert!(self.columns() == rhs.rows(), "Mismatched Matrices");

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

    fn add(self, rhs: Self) -> Self::Output {
        // LxM + LxM -> LxM
        assert!(self.size() == rhs.size(), "Mismatched Matrices");

        let mut result = Matrix::empty(self.get(0, 0), self.size());
        for i in 0..self.rows() {
            for j in 0..self.columns() {
                result.set(i, j, self.get_reference(i, j) + rhs.get_reference(i, j));
            }
        }
        result
    }
}

/**
 * Elementwise addition for two matrices.
 */
impl <T: Numeric> Add for Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Matrix<T>;
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

    fn sub(self, rhs: Self) -> Self::Output {
        // LxM - LxM -> LxM
        assert!(self.size() == rhs.size(), "Mismatched Matrices");

        let mut result = Matrix::empty(self.get(0, 0), self.size());
        for i in 0..self.rows() {
            for j in 0..self.columns() {
                result.set(i, j, self.get_reference(i, j) - rhs.get_reference(i, j));
            }
        }
        result
    }
}

/**
 * Elementwise subtraction for two matrices.
 */
impl <T: Numeric> Sub for Matrix<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Matrix<T>;
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

    fn neg(self) -> Self::Output {
        - &self
    }
}
