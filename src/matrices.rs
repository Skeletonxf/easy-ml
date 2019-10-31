use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Div;
use std::ops::Neg;
use std::iter::Sum;
use std::cmp::PartialOrd;
use std::marker::Sized;

/**
 * A general purpose numeric trait that defines all the behaviour numerical matrices need
 * their types to support for math operations.
 */
pub trait Numeric: Add + Sub + Mul + Div + Neg + Sum + PartialOrd + Sized + Clone {}

/**
 * Anything which implements all the super traits will automatically implement this trait too.
 * This covers primitives such as f32, f64.
 */
impl<T: Add + Sub + Mul + Div + Neg + Sum + PartialOrd + Sized + Clone> Numeric for T {}

/**
 * A general purpose matrix of some type.
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
}

/**
 * An iterator over a column in a matrix.
 */
pub struct ColumnIterator<'a, T: Clone> {
    matrix: &'a Matrix<T>,
    column: Column,
    counter: usize,
}

impl <'a, T: Clone> Iterator for ColumnIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: bounds checking for counter == usize
        self.counter += 1;

        if self.counter > self.matrix.rows() {
            None
        } else {
            Some(self.matrix.data[self.counter - 1][self.column].clone())
        }
    }
}

/**
 * An iterator over a row in a matrix.
 */
pub struct RowIterator<'a, T: Clone> {
    matrix: &'a Matrix<T>,
    row: Row,
    counter: usize,
}

impl <'a, T: Clone> Iterator for RowIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: bounds checking for counter == usize
        self.counter += 1;

        if self.counter > self.matrix.columns() {
            None
        } else {
            Some(self.matrix.data[self.row][self.counter - 1].clone())
        }
    }
}

/**
 * Methods for matrices with types that can be copied.
 */
impl <T: Clone> Matrix<T> {
    /**
     * Computes and returns the transpose of this matrix
     */
    pub fn transpose(&self) -> Matrix<T> {
        let mut result = Matrix { data: Vec::new() };
        for i in 0..self.columns() {
            result.data.push(Vec::new());
            for j in 0..self.rows() {
                result.data[i].push(self.data[j][i].clone());
            }
        }
        result
    }

    /**
     * Returns an iterator over a column vector in this matrix. Columns are 0 indexed.
     */
    pub fn column_iter(&self, column: Column) -> ColumnIterator<T> {
        ColumnIterator {
            matrix: &self,
            column,
            counter: 0,
        }
    }

    /**
     * Returns an iterator over a row vector in this matrix. Rows are 0 indexed.
     */
    pub fn row_iter(&self, row: Row) -> RowIterator<T> {
        RowIterator {
            matrix: &self,
            row,
            counter: 0,
        }
    }
}

/**
 * Methods for matrices with numerical types, such as f32 or f64
 */
impl <T: Numeric> Matrix<T> {
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
 */
impl <T: Numeric> Mul for &Matrix<T> where T: Mul<Output = T> {
    // Tell the compiler our output type is another matrix of type T
    type Output = Matrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        // LxM * MxN -> LxN

        assert!(self.columns() == rhs.rows(), "Mismatched Matrices");

        let mut result = Matrix { data: Vec::new() };
        for i in 0..self.rows() {
            result.data.push(Vec::new());
            for j in 0..rhs.columns() {
                // compute dot product for each element in the new matrix
                result.data[i].push(self.row_iter(i)
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
