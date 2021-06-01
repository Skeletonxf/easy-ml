/*!
 * Eigenvalues and eigenvectors definitions and solvers.
 */

use crate::linear_algebra;
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
     * If the input is not diagonalizable the eigendecomposition will also fail regardless of
     * the algorithm because only diagonalizable matrices can be factorized in this way.
     * For example, a matrix of [[1, 1], [0, 1]] is [defective](https://en.wikipedia.org/wiki/Defective_matrix),
     * and no eigenvalue decomposition solution exists for it. It is not specified how an
     * [EigenvalueAlgorithm] solver should deal with such non diagonalizable inputs, ideally they
     * would detect the input has no solution and return an Err variant, but they may loop
     * infinitely or panic instead.
     */
    fn solve(&mut self, matrix: &Matrix<T>) -> Result<Eigens<T>, EigenvalueAlgorithmError>;
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
    InvalidInput(Box<dyn error::Error>),
    /**
     * Some generic failiure.
     */
    Failed(Box<dyn error::Error>),
}

pub trait ConvergenceCriteria<T: Numeric + Real>
where for<'a> &'a T: NumericRef<T> + RealRef<T> {
    fn is_converged(&mut self, matrix: &Matrix<T>) -> bool;
}

pub struct FixedIterations {
    remaining: u64,
}

impl FixedIterations {
    pub fn new(count: u64) -> FixedIterations {
        FixedIterations {
            remaining: count,
        }
    }
}

impl<T: Numeric + Real> ConvergenceCriteria<T> for FixedIterations
where for<'a> &'a T: NumericRef<T> + RealRef<T> {
    fn is_converged(&mut self, _: &Matrix<T>) -> bool {
        self.remaining = self.remaining.saturating_sub(1);
        self.remaining == 0
    }
}

pub struct QRAlgorithm<C> {
    convergence: C,
    _private: (),
}

impl <C> QRAlgorithm<C> {
    pub fn new<T>(convergence: C) -> QRAlgorithm<C>
    where
        T: Numeric + Real,
        for<'a> &'a T: NumericRef<T> + RealRef<T>,
        C: ConvergenceCriteria<T>,
    {
        QRAlgorithm {
            convergence,
            _private: (),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct UnableToDecomposeMatrix;

impl fmt::Display for UnableToDecomposeMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QR decomposition failed on the input matrix or a later iteration.")
    }
}

impl error::Error for UnableToDecomposeMatrix {}

impl<C, T> EigenvalueAlgorithm<T> for QRAlgorithm<C>
where
    T: Numeric + Real,
    for<'a> &'a T: NumericRef<T> + RealRef<T>,
    C: ConvergenceCriteria<T>,
{
    // Please don't use this yet, this unshifted QR algorithm is arbitrarily slow and not remotely
    // efficient enough to use on any size input. It's also not likely to converge, and
    // fails on inputs like [0, 1; 1, 0]
    fn solve(&mut self, matrix: &Matrix<T>) -> Result<Eigens<T>, EigenvalueAlgorithmError> {
        // TODO: A 1x1 input should be special cased
        // TODO: Check the matrix is square
        // References
        // http://pi.math.cornell.edu/~web6140/TopTenAlgorithms/QRalgorithm.html
        // https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf
        // https://www-users.cs.umn.edu/~saad/csci5304/FILES/LecN13.pdf
        // https://madrury.github.io/jekyll/update/statistics/2017/10/04/qr-algorithm.html
        // https://maxwell.ict.griffith.edu.au/spl/publications/papers/QRPCA_IJMLC.pdf
        let n = matrix.rows();
        let mut u = Matrix::diagonal(T::one(), (n, n));
        let qr = linear_algebra::qr_decomposition::<T>(matrix)
            .ok_or(EigenvalueAlgorithmError::InvalidInput(Box::new(UnableToDecomposeMatrix)))?;
        let (q, r) = (qr.q, qr.r);
        let mut a = r * &q;
        u = u * q;
        loop {
            if self.convergence.is_converged(&a) {
                break;
            }
            // TODO: Wilkinson shifts
            let qr = linear_algebra::qr_decomposition::<T>(&a)
                .ok_or(EigenvalueAlgorithmError::InvalidInput(Box::new(UnableToDecomposeMatrix)))?;
            let (q, r) = (qr.q, qr.r);
            a = r * &q;
            u = u * q;
        }
        // read the eigenvalues off the diagonal
        let eigenvalues = {
            let mut eigenvalues = Vec::with_capacity(a.rows());
            for i in 0..a.rows() {
                eigenvalues.push(a.get(i, i));
            }
            eigenvalues
        };
        let eigenvectors = u;
        Eigens::new(eigenvalues, eigenvectors)
    }
}
