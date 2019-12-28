/*!
 * Generic matrix type
 */

use std::ops::{Add, Sub, Mul, Neg, Div};

pub mod iterators;

use crate::matrices::iterators::{ColumnIterator, RowIterator, ColumnMajorIterator};
use crate::numeric::Numeric;
use crate::linear_algebra;

/**
 * A general purpose matrix of some type. This type may implement
 * no traits, in which case the matrix will be rather useless. If the
 * type implements [`Clone`](https://doc.rust-lang.org/std/clone/trait.Clone.html)
 * most storage and accessor methods are defined and if the type implements
 * [`Numeric`](../numeric/trait.Numeric.html) then the matrix can be used in
 * a mathematical way.
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
     * being a row, and hence the outer vector containing all rows in sequence
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
}

/**
 * Methods for matrices with types that can be copied, but still not neccessarily numerical.
 */
impl <T: Clone> Matrix<T> {
    /**
     * Computes and returns the transpose of this matrix
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
     */
    pub fn column_iter(&self, column: Column) -> ColumnIterator<T> {
        ColumnIterator::new(self, column)
    }

    /**
     * Returns an iterator over a row vector in this matrix. Rows are 0 indexed.
     */
    pub fn row_iter(&self, row: Row) -> RowIterator<T> {
        RowIterator::new(self, row)
    }

    /**
     * Returns a column major iterator over all values in this matrix, proceeding through each
     * column in order.
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
     * the matrix.
     */
    pub fn map_mut(&mut self, mapping_function: impl Fn(T) -> T) {
        for i in 0..self.rows() {
            for j in 0..self.columns() {
                self.set(i, j, mapping_function(self.get(i, j).clone()));
            }
        }
    }

    /**
     * Creates and returns a new matrix with all values from the original with the
     * function applied to each.
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
     * Inserts a new row into the Matrix immediately after the provided index,
     * shifting other rows to the right and filling all entries with the
     * provided value. Rows are 0 indexed.
     *
     * This will panic if the row is greater than the number of rows in the matrix.
     */
    pub fn insert_row(&mut self, row: Row, value: T) {
        let new_row = vec![value.clone(); self.columns()];
        self.data.insert(row, new_row);
    }

    /**
     * Inserts a new column into the Matrix immediately after the provided index,
     * shifting other columns to the right and filling all entries with the
     * provided value. Columns are 0 indexed.
     *
     * This will panic if the column is greater than the number of columns in the matrix.
     */
    pub fn insert_column(&mut self, column: Column, value: T) {
        for row in 0..self.rows() {
            self.data[row].insert(column, value.clone());
        }
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
 * While these will all be defined on signed integer types as well, such as i16 or i32,
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
impl <T: Numeric> Matrix<T> {
    /**
     * Creates an identity matrix of the provided size. An identity matrix
     * is always square and has elements equal to 1 along its diagonal and
     * zero everywhere else. The size is still taken as a tuple to facilitate
     * creating an identity matrix from the dimensionality of an existing one.
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
    pub fn identity(size: (Row, Column)) -> Matrix<T> {
        assert!(size.0 == size.1);
        let mut matrix = Matrix {
            data: vec![vec![T::zero(); size.1]; size.0]
        };
        for i in 0..size.0 {
            matrix.set(i, i, T::one());
        }
        matrix
    }

    /**
     * Returns the determinant of this square matrix, or None if the matrix
     * does not have a determinant. See [`linear_algebra`](../linear_algebra/fn.determinant.html)
     */
    pub fn determinant(&self) -> Option<T>
    where T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> {
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
}

/**
 * PartialEq is implemented as two matrices are equal if and only if all their elements
 * are equal and they have the same size.
 */
impl <T: Numeric> PartialEq for Matrix<T> {
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
impl <T: Numeric> Mul for &Matrix<T> where T: Mul<Output = T> {
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
                    self.row_iter(i)
                    .zip(rhs.column_iter(j))
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
impl <T: Numeric> Mul for Matrix<T> where T: Mul<Output = T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

/**
 * Matrix multiplication for two matrices with one referenced.
 */
impl <T: Numeric> Mul<&Matrix<T>> for Matrix<T> where T: Mul<Output = T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

/**
 * Matrix multiplication for two matrices with one referenced.
 */
impl <T: Numeric> Mul<Matrix<T>> for &Matrix<T> where T: Mul<Output = T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        self * &rhs
    }
}

/**
 * Elementwise addition for two referenced matrices.
 */
impl <T: Numeric> Add for &Matrix<T> where T: Add<Output = T> {
    // Tell the compiler our output type is another matrix of type T
    type Output = Matrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        // LxM + LxM -> LxM
        assert!(self.size() == rhs.size(), "Mismatched Matrices");

        let mut result = Matrix::empty(self.get(0, 0), self.size());
        for i in 0..self.rows() {
            for j in 0..self.columns() {
                // compute dot product for each element in the new matrix
                result.set(i, j, self.get(i, j) + rhs.get(i, j));
            }
        }
        result
    }
}

/**
 * Elementwise addition for two matrices.
 */
impl <T: Numeric> Add for Matrix<T> where T: Add<Output = T> {
    type Output = Matrix<T>;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

/**
 * Elementwise addition for two matrices with one referenced.
 */
impl <T: Numeric> Add<&Matrix<T>> for Matrix<T> where T: Add<Output = T> {
    type Output = Matrix<T>;
    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

/**
 * Elementwise addition for two matrices with one referenced.
 */
impl <T: Numeric> Add<Matrix<T>> for &Matrix<T> where T: Add<Output = T> {
    type Output = Matrix<T>;
    fn add(self, rhs: Matrix<T>) -> Self::Output {
        self + &rhs
    }
}

/**
 * Elementwise subtraction for two referenced matrices.
 */
impl <T: Numeric> Sub for &Matrix<T> where T: Sub<Output = T> {
    // Tell the compiler our output type is another matrix of type T
    type Output = Matrix<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        // LxM - LxM -> LxM
        assert!(self.size() == rhs.size(), "Mismatched Matrices");

        let mut result = Matrix::empty(self.get(0, 0), self.size());
        for i in 0..self.rows() {
            for j in 0..self.columns() {
                // compute dot product for each element in the new matrix
                result.set(i, j, self.get(i, j) - rhs.get(i, j));
            }
        }
        result
    }
}

/**
 * Elementwise subtraction for two matrices.
 */
impl <T: Numeric> Sub for Matrix<T> where T: Sub<Output = T> {
    type Output = Matrix<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

/**
 * Elementwise subtraction for two matrices with one referenced.
 */
impl <T: Numeric> Sub<&Matrix<T>> for Matrix<T> where T: Sub<Output = T> {
    type Output = Matrix<T>;
    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}

/**
 * Elementwise subtraction for two matrices with one referenced.
 */
impl <T: Numeric> Sub<Matrix<T>> for &Matrix<T> where T: Sub<Output = T> {
    type Output = Matrix<T>;
    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        self - &rhs
    }
}

/**
 * Elementwise negation for a referenced matrix.
 */
impl <T: Numeric> Neg for &Matrix<T> where T: Neg<Output = T> {
    // Tell the compiler our output type is another matrix of type T
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        self.map(|v| -v)
    }
}

/**
 * Elementwise negation for a matrix.
 */
impl <T: Numeric> Neg for Matrix<T> where T: Neg<Output = T> {
    // Tell the compiler our output type is another matrix of type T
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        - &self
    }
}
