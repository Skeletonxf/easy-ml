/*!
 * Linear algebra algorithms on numbers and matrices
 *
 * Note that these functions are also exposed as corresponding methods on the Matrix type,
 * but in depth documentation is only presented here.
 *
 * It is recommended to favor the corresponding methods on the Matrix type as the
 * Rust compiler can get confused with the generics on these functions if you use
 * these methods without turbofish syntax.
 */

use crate::matrices::{Matrix, Row, Column};
use crate::numeric::{Numeric, NumericRef};
use crate::numeric::extra::Sqrt;

/**
 * Computes the inverse of a matrix provided that it exists. To have an inverse
 * a matrix must be square (same number of rows and columns) and it must also
 * have a non zero determinant.
 *
 * The inverse of a matrix `A` is the matrix `A^-1` which when multiplied by `A`
 * in either order yields the identity matrix `I`.
 *
 * `A(A^-1) == (A^-1)A == I`.
 *
 *The inverse is like the reciprocal of a number, except for matrices instead of scalars.
 * With scalars, there is no inverse for `0` because `1 / 0` is not defined. Similarly
 * to compute the inverse of a matrix we divide by the determinant, so matrices
 * with a determinant of 0 have no inverse, even if they are square.
 *
 * This algorithm performs the analytic solution described by
 * [wikipedia](https://en.wikipedia.org/wiki/Invertible_matrix#Analytic_solution)
 * and should compute the inverse for any size of square matrix if it exists, but
 * is inefficient for large matrices.
 *
 * # Warning
 *
 * With some uses of this function the Rust compiler gets confused about what type `T`
 * should be and you will get the error:
 * > overflow evaluating the requirement `&'a _: easy_ml::numeric::NumericByValue<_, _>`
 *
 * In this case you need to manually specify the type of T by using the
 * turbofish syntax like:
 * `linear_algebra::inverse::<f32>(&matrix)`
 *
 * Alternatively, the compiler doesn't seem to run into this problem if you
 * use the equivalent methods on the matrix type like so:
 * `matrix.inverse()`
 */
pub fn inverse<T: Numeric>(matrix: &Matrix<T>) -> Option<Matrix<T>>
where for<'a> &'a T: NumericRef<T> {
    if matrix.rows() != matrix.columns() {
        return None;
    }
    // inverse of a 1 x 1 matrix is a special case
    if matrix.rows() == 1 {
        // determinant of a 1 x 1 matrix is the single element
        let element = matrix.get(0, 0);
        if element == T::zero() {
            return None;
        }
        return Some(Matrix::unit(T::one() / element));
    }

    // compute the general case for a N x N matrix where N >= 2
    match determinant(matrix) {
        Some(det) => {
            if det == T::zero() {
                return None;
            }
            let determinant_reciprocal = T::one() / det;
            let mut cofactor_matrix = Matrix::empty(T::zero(), matrix.size());
            for i in 0..matrix.rows() {
                for j in 0..matrix.columns() {
                    // this should always return Some due to the earlier checks
                    let ij_minor = minor(matrix, i, j)?;
                    // i and j may each be up to the maximum value for usize but
                    // we only need to know if they are even or add as
                    // -1 ^ (i + j) == -1 ^ ((i % 2) + (j % 2))
                    // by taking modulo of both before adding we ensure there
                    // is no overflow
                    let sign = i8::pow(-1, (i.rem_euclid(2) + j.rem_euclid(2)) as u32);
                    // convert sign into type T
                    let sign = if sign == 1 {
                        T::one()
                    } else {
                        T::zero() - T::one()
                    };
                    // each element of the cofactor matrix is -1^(i+j) * M_ij
                    // for M_ij equal to the ij minor of the matrix
                    cofactor_matrix.set(i, j, sign * ij_minor);
                }
            }
            // tranposing the cofactor matrix yields the adjugate matrix
            cofactor_matrix.transpose_mut();
            // finally to compute the inverse we need to multiple each element by 1 / |A|
            cofactor_matrix.map_mut(|element| element * determinant_reciprocal.clone());
            Some(cofactor_matrix)
        },
        None => None
    }
}

// TODO: expose these minor methods and test them directly
// https://www.teachoo.com/9780/1204/Minor-and-Cofactor-of-a-determinant/category/Finding-Minors-and-cofactors/

/*
 * Computes the (i,j) minor of a matrix by copying it. This is the
 * determinant of the matrix after deleting the ith row and the jth column.
 *
 * Minors can only be taken on matrices which have a determinant and rows and
 * columns to remove. Hence for non square matrices or 1 x 1 matrices this returns
 * None.
 */
fn minor<T: Numeric>(matrix: &Matrix<T>, i: Row, j: Column) -> Option<T>
where for<'a> &'a T: NumericRef<T> {
    minor_mut(&mut matrix.clone(), i, j)
}

/**
 * Computes the (i,j) minor of a matrix by modifying it in place. This is
 * the determinant of the matrix after deleting the ith row and the jth column.
 *
 * Minors can only be taken on matrices which have a determinant and rows and
 * columns to remove. Hence for non square matrices or 1 x 1 matrices this returns
 * None and does not modify the matrix.
 */
fn minor_mut<T: Numeric>(matrix: &mut Matrix<T>, i: Row, j: Column) -> Option<T>
where for<'a> &'a T: NumericRef<T> {
    if matrix.rows() == 1 || matrix.columns() == 1 {
        // nothing to delete
        return None;
    }
    if matrix.rows() != matrix.columns() {
        // no determinant
        return None;
    }
    matrix.remove_row(i);
    matrix.remove_column(j);
    determinant(matrix)
}

/**
 * Computes the determinant of a square matrix. For a 2 x 2 matrix this is given by
 * `ad - bc` for:
 * ```ignore
 * [
 *   a, b
 *   c, d
 * ]
 * ```
 *
 * This function will return the determinant only if it exists. Non square matrices
 * do not have a determinant. A determinant is a scalar value computed from the
 * elements of a square matrix and often corresponds to matrices with special properties.
 *
 * Note that the determinant of a 1 x 1 matrix is just the element in the matrix.
 *
 * This function computes the determinant using the same type as that of the Matrix,
 * hence if the input type is unsigned (such as Wrapping&lt;u8&gt;) the value computed
 * is likely to not make any sense because a determinant may be negative.
 *
 * [https://en.wikipedia.org/wiki/Determinant](https://en.wikipedia.org/wiki/Determinant)
 *
 * # Warning
 *
 * With some uses of this function the Rust compiler gets confused about what type `T`
 * should be and you will get the error:
 * > overflow evaluating the requirement `&'a _: easy_ml::numeric::NumericByValue<_, _>`
 *
 * In this case you need to manually specify the type of T by using the
 * turbofish syntax like:
 * `linear_algebra::determinant::<f32>(&matrix)`
 *
 * Alternatively, the compiler doesn't seem to run into this problem if you
 * use the equivalent methods on the matrix type like so:
 * `matrix.determinant()`
 */
pub fn determinant<T: Numeric>(matrix: &Matrix<T>) -> Option<T>
where for<'a> &'a T: NumericRef<T> {
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let length = matrix.rows();

    if length == 0 {
        return None;
    }

    if length == 1 {
        return Some(matrix.get(0, 0));
    }

    // compute the general case for the determinant of an N x N matrix with
    // N >= 2

    let mut sum = T::zero();

    // iterate through all permutations of the numbers in the range from 0 to N - 1
    // which we will use for indexing
    with_each_permutation(&mut (0..length).collect(), &mut |permutation, even_swap| {
        // Compute the signature for this permutation, such that we
        // have +1 for an even number and -1 for an odd number of swaps
        let signature = if even_swap {
            T::one()
        } else {
            T::zero() - T::one()
        };
        let mut product = T::one();
        for (n, i) in permutation.iter().enumerate() {
            // Get the element at the index corresponding to n and the n'th
            // element in the permutation list.
            let element = matrix.get_reference(n, *i);
            product = product * element;
        }
        // copying the sum to prevent a move that stops us from returning it
        // still massively reduces the amount of copies compared to using
        // generate_permutations which would instead require copying the
        // permutation list N! times though allow to not copy the sum.
        sum = sum.clone() + (signature * product);
    });

    Some(sum)
}

/*
 * Computes the factorial of a number.
 * eg for an input of 5 computes 1 * 2 * 3 * 4 * 5
 * which is equal to 120
 */
#[allow(dead_code)] // used in testing
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/**
 * Performs repeated swaps on the provided mutable reference to a list, swapping
 * exactly 1 pair each time before calling the consumer as defined by Heap's Algorithm
 * https://en.wikipedia.org/wiki/Heap%27s_algorithm
 */
fn heaps_permutations<T: Clone, F>(k: usize, list: &mut Vec<T>, consumer: &mut F)
where F: FnMut(&mut Vec<T>) {
    if k == 1 {
        consumer(list);
        return;
    }

    for i in 0..k {
        heaps_permutations(k - 1, list, consumer);
        // avoid redundant swaps
        if i < k - 1 {
            // Swap on the even/oddness of k
            if k % 2 == 0 {
                // if k is even swap final and the index
                list.swap(i, k - 1);
            } else {
                // if k is odd swap final and first
                list.swap(0, k - 1);
            }
        }
    }
}

/**
 * Generates a list of all possible permutations of a list, with each
 * sublist one swap different from the last and correspondingly alternating
 * in even and odd swaps required to obtain the reordering.
 */
#[allow(dead_code)] // used in testing
fn generate_permutations<T: Clone>(list: &mut Vec<T>) -> Vec<(Vec<T>, bool)> {
    let mut permutations = Vec::with_capacity(factorial(list.len()));
    let mut even_swaps = true;
    heaps_permutations(list.len(), list, &mut |permuted| {
        permutations.push((permuted.clone(), even_swaps));
        even_swaps = !even_swaps;
    });
    permutations
}

/*
 * Inplace version of generate_permutations which calls the consumer on
 * each permuted list without performing any copies (ie each permuted list)
 * is the same list before and after permutation.
 */
fn with_each_permutation<T: Clone, F>(list: &mut Vec<T>, consumer: &mut F)
where F: FnMut(&mut Vec<T>, bool) {
    let mut even_swaps = true;
    heaps_permutations(list.len(), list, &mut |permuted| {
        consumer(permuted, even_swaps);
        even_swaps = !even_swaps;
    });
}

#[test]
fn test_permutations() {
    // Exhaustively test permutation even/oddness for an input
    // of length 3
    let mut list = vec![ 1, 2, 3 ];
    let permutations = generate_permutations(&mut list);
    assert!(permutations.contains(&(vec![1, 2, 3], true)));
    assert!(permutations.contains(&(vec![3, 2, 1], false)));
    assert!(permutations.contains(&(vec![2, 3, 1], true)));
    assert!(permutations.contains(&(vec![1, 3, 2], false)));
    assert!(permutations.contains(&(vec![2, 1, 3], false)));
    assert!(permutations.contains(&(vec![3, 1, 2], true)));
    assert_eq!(permutations.len(), 6);

    // Test a larger input non exhaustively to make sure it
    // generalises.
    let mut list = vec![ 1, 2, 3, 4, 5 ];
    let permuted = generate_permutations(&mut list);
    assert!(permuted.contains(&(vec![1, 2, 3, 4, 5], true)));
    assert!(permuted.contains(&(vec![1, 2, 3, 5, 4], false)));
    assert!(permuted.contains(&(vec![1, 2, 5, 3, 4], true)));

    // Test a length 2 input as well
    let mut list = vec![0, 1];
    let permuted = generate_permutations(&mut list);
    assert!(permuted.contains(&(vec![0, 1], true)));
    assert!(permuted.contains(&(vec![1, 0], false)));
    assert_eq!(permuted.len(), 2);
}

/**
 * Computes the covariance matrix for an NxM feature matrix, in which
 * each N'th row has M features to find the covariance and variance of.
 *
 * Each element in the covariance matrix at (i, j) will be the variance of the
 * ith and jth features from the feature matrix, defined as the zero meaned
 * dot product of the two feature vectors divided by the number of samples.
 *
 * If the numeric type is unable to represent the number of samples
 * for each feature (ie if `T: u8` and you have 1000 samples) then this function
 * will return `None`. Otherwise the covariance matrix will always be
 * computed and the function will return `Some`.
 *
 * # Warning
 *
 * With some uses of this function the Rust compiler gets confused about what type `T`
 * should be and you will get the error:
 * > overflow evaluating the requirement `&'a _: easy_ml::numeric::NumericByValue<_, _>`
 *
 * In this case you need to manually specify the type of T by using the
 * turbofish syntax like:
 * `linear_algebra::covariance::<f32>(&matrix)`
 *
 * Alternatively, the compiler doesn't seem to run into this problem if you
 * use the equivalent methods on the matrix type like so:
 * `matrix.covariance()`
 */
// TODO: mention Bessel's correction, do covariance and correlation and explain
// difference, do DataxFeatures and FeaturesxData
pub fn covariance<T: Numeric>(matrix: &Matrix<T>) -> Option<Matrix<T>>
where for<'a> &'a T: NumericRef<T> {
    let features = matrix.columns();
    let samples = T::from_usize(matrix.rows())?;
    let mut covariance_matrix = Matrix::empty(T::zero(), (features, features));
    covariance_matrix.map_mut_with_index(|_, i, j| {
        // set each element of the covariance matrix to the variance
        // of features i and j
        let feature_i_mean: T = matrix.column_iter(i).sum::<T>() / &samples;
        let feature_j_mean: T = matrix.column_iter(j).sum::<T>() / &samples;
        matrix.column_reference_iter(i)
            .map(|x| x - &feature_i_mean)
            .zip(matrix.column_reference_iter(j).map(|y| y - &feature_j_mean))
            .map(|(x, y)| x * y)
            .sum::<T>() / &samples
    });
    Some(covariance_matrix)
}

/**
 * Computes the cholesky decomposition of a matrix. This yields a matrix `L`
 * such that for the provided matrix `A`, `L * L^T = A`. `L` will always be
 * lower triangular, ie all entries above the diagonal will be 0. Hence cholesky
 * decomposition can be interpreted as a generalised square root function.
 *
 * Cholesky decomposition is defined for
 * [Hermitian](https://en.wikipedia.org/wiki/Hermitian_matrix),
 * [positive definite](https://en.wikipedia.org/wiki/Definiteness_of_a_matrix)
 * matrices. For a real valued (ie not containing complex numbers) matrix, if it is
 * [Symmetric](https://en.wikipedia.org/wiki/Symmetric_matrix) it is Hermitian.
 *
 * This function does not check that the provided matrix is Hermitian or
 * positive definite at present, but may in the future, in which case `None`
 * will be returned.
 */
pub fn cholesky_decomposition<T: Numeric + Sqrt<Output = T>>(matrix: &Matrix<T>) -> Option<Matrix<T>>
where for<'a> &'a T: NumericRef<T> {
    if matrix.rows() != matrix.columns() {
        return None;
    }
    // The computation steps are outlined nicely at https://rosettacode.org/wiki/Cholesky_decomposition
    let mut lower_triangular = Matrix::empty(T::zero(), matrix.size());
    for i in 0..lower_triangular.rows() {
        // For each column k we need to compute all i, k entries
        // before incrementing k further as the diagonals depend
        // on the elements below the diagonal of the previous columns,
        // and the elements below the diagonal depend on the diagonal
        // of their column and elements below the diagonal up to that
        // column.
        for k in 0..lower_triangular.columns() {
            if i == k {
                let mut sum = T::zero();
                for j in 0..k {
                    sum = &sum + (lower_triangular.get_reference(k, j)
                        * lower_triangular.get_reference(k, j));
                }
                lower_triangular.set(i, k, (matrix.get_reference(i, k) - sum).sqrt());
            }
            // after 0,0 we iterate down the rows for i,k where i > k
            // these elements only depend on values already computed in
            // their own column and prior columns
            if i > k {
                let mut sum = T::zero();
                for j in 0..k {
                    sum = &sum + (lower_triangular.get_reference(i, j)
                        * lower_triangular.get_reference(k, j));
                }
                lower_triangular.set(i, k,
                    (T::one() / lower_triangular.get_reference(k, k)) *
                    (matrix.get_reference(i, k) - sum));
            }
        }
    }
    Some(lower_triangular)
}
