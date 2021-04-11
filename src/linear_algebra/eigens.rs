/*!
 * Eigenvalues and eigenvectors definitions and solvers.
 */

use crate::matrices::Matrix;
use crate::numeric::{Numeric, NumericRef};
use crate::numeric::extra::{Real, RealRef};

use std::fmt;
use std::error;

/**
 * The interface for Eigenvalue Algorithms.
 */
pub trait EigenvalueAlgorithm<T: Numeric + Real>
where for<'a> &'a T: NumericRef<T> + RealRef<T> {
    /**
     * For an input matrix of size NxN, computes the N eigenvalue and eigenvector pairs of the
     * input
     *
     * This is usually an iterative algorithm which could fail to converge.
     */
    fn solve(&self, matrix: &Matrix<T>) -> Result<Eigens<T>, EigenvalueAlgorithmError>;
}

pub struct Eigens<T> {
    pub eigenvalues: Vec<T>,
    pub eigenvectors: Matrix<T>,
    _private: (),
}

impl <T> Eigens<T> {
    /**
     * Creates an Eigens struct from a list of eigenvalues and their corresponding eigenvectors.
     *
     * Each ith eigenvalue must correspond to the ith eigencolumnvector in the eigenvectors input.
     *
     * The eigenvalues and eigenvector pairs will be automatically sorted so that they are in
     * decreasing eigenvalue order.
     */
    pub fn new(eigenvalues: Vec<T>, eigenvectors: Matrix<T>) -> Result<Eigens<T>, EigenvalueAlgorithmError> {
        if eigenvectors.rows() != eigenvectors.columns() {
            Err(EigenvalueAlgorithmError::InvalidInput(Box::new(EigenvectorsNotSquare {
                size: eigenvectors.size()
            })))
        } else if eigenvalues.len() != eigenvectors.rows() {
            Err(EigenvalueAlgorithmError::InvalidInput(Box::new(EigensMismatched {
                values: eigenvalues.len(),
                vectors: eigenvectors.size()
            })))
        } else {
            // TODO: Sort input
            Ok(Eigens {
                eigenvalues,
                eigenvectors,
                _private: (),
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EigenvectorsNotSquare {
    size: (usize, usize),
}

impl fmt::Display for EigenvectorsNotSquare {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Eigenvectors matrix is not square: {}x{}", self.size.0, self.size.1)
    }
}

impl error::Error for EigenvectorsNotSquare {}

#[derive(Clone, Debug, PartialEq)]
pub struct EigensMismatched {
    values: usize,
    vectors: (usize, usize),
}

impl fmt::Display for EigensMismatched {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Eigenvalues and eigenvectors matrix are not the same length: {} vs {}x{}", self.values, self.vectors.0, self.vectors.1)
    }
}

impl error::Error for EigensMismatched {}

/**
 * An enumeration of reasons an [EigenvalueAlgorithm] solver may have failed.
 */
#[non_exhaustive]
pub enum EigenvalueAlgorithmError {
    /**
     * The input was invalid or is unsupported by this solver.
     */
    InvalidInput(Box<dyn std::error::Error>),
    /**
     * Some generic failiure.
     */
    Failed(Box<dyn std::error::Error>),
}
