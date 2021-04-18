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
     * For a [diagonalizable](https://en.wikipedia.org/wiki/Diagonalizable_matrix) input matrix
     * of size NxN, computes the N eigenvalue and eigenvector pairs of the input
     *
     * This is usually an iterative algorithm which could fail to converge.
     *
     *If the input is not diagonalizable the eigendecomposition will also fail regardless of
     * the algorithm because only diagonalizable matrices can be factorized in this way.
     * For example, a matrix of [[1, 1], [0, 1]] is [defective](https://en.wikipedia.org/wiki/Defective_matrix),
     * and no eigenvalue decomposition solution exists for it. It is not specified how an
     * [EigenvalueAlgorithm] solver should deal with such non diagonalizable inputs, ideally they
     * would detect the input has no solution and return an Err variant, but they may loop
     * infinitely or panic instead.
     */
    fn solve(&self, matrix: &Matrix<T>) -> Result<Eigens<T>, EigenvalueAlgorithmError>;
}

/**
 * The eigendecomposition of a matrix.
 *
 * For a [diagonalizable](https://en.wikipedia.org/wiki/Diagonalizable_matrix) square matrix A of
 * size NxN, N eigenvalue and eigenvector pairs can be computed.
 *
 * Eigenvalues and eigenvectors have many proprties, but focusing on eigenvalue decomposition,
 * you can construct a matrix Q of size NxN with its N columns corresponding to each eigenvector
 * of A and a diagonal matrix Λ of size NxN with the N diagonal entries corresponding to each
 * eigenvalue of A (keeping the pairs of eigenvalues and eigenvectors together).
 *
 * Then: A = Q Λ Q<sup>-1</sup> and Λ = Q<sup>-1</sup> A Q
 */
pub struct Eigens<T> {
    /**
     * The N eigenvalues. Each eigenvalue is paired with the eigenvector in the same
     * column of the eigenvectors as the eigenvalue's index.
     */
    pub eigenvalues: Vec<T>,
    /**
     * The NxN matrix of N eigenvectors, stored in the N columns of the matrix. Each
     * eigenvector is paired with the eigenvalue in the same index of the eigenvalues
     * as the eigenvector's column.
     */
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

/**
 * Not a square matrix.
 */
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

/**
 * There were not the same number of eigenvalues as eigenvectors.
 */
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
