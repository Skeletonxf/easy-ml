/*!
 * Linear algebra algorithms on numbers and matrices
 *
 * Note that many of these functions are also exposed as corresponding methods on the Matrix type,
 * and the Tensor type, but in depth documentation is only presented here.
 *
 * It is recommended to favor the corresponding methods on the Matrix and Tensor types as the
 * Rust compiler can get confused with the generics on these functions if you use
 * these methods without turbofish syntax.
 *
 * Nearly all of these functions are generic over [Numeric](super::numeric) types,
 * unfortunately, when using these functions the compiler may get confused about what
 * type `T` should be and you will get the error:
 * > overflow evaluating the requirement `&'a _: easy_ml::numeric::NumericByValue<_, _>`
 *
 * In this case you need to manually specify the type of T by using the
 * turbofish syntax like:
 * `linear_algebra::inverse::<f32>(&matrix)`
 *
 * You might be working with a generic type of T, in which case specify that
 * `linear_algebra::inverse::<T>(&matrix)`
 *
 * ## Generics
 *
 * For the tensor variants of these functions, the generics allow very flexible input types.
 *
 * A function like
 * ```ignore
 * pub fn inverse_tensor<T, S, I>(tensor: I) -> Option<Tensor<T, 2>> where
 *    T: Numeric,
 *    for<'a> &'a T: NumericRef<T>,
 *    I: Into<TensorView<T, S, 2>>,
 *    S: TensorRef<T, 2>,
 * ```
 * Means it takes any type that can be converted to a TensorView, which includes Tensor, &Tensor,
 * &mut Tensor as well as references to a TensorView.
 */

use crate::matrices::{Column, Matrix, Row};
use crate::numeric::extra::{Real, RealRef, Sqrt};
use crate::numeric::{Numeric, NumericRef};
use crate::tensors::views::{TensorRef, TensorView};
use crate::tensors::{Dimension, Tensor};

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
where
    for<'a> &'a T: NumericRef<T>,
{
    if matrix.rows() != matrix.columns() {
        return None;
    }
    // inverse of a 1 x 1 matrix is a special case
    if matrix.rows() == 1 {
        // determinant of a 1 x 1 matrix is the single element
        let element = matrix.scalar();
        if element == T::zero() {
            return None;
        }
        return Some(Matrix::from_scalar(T::one() / element));
    }

    // compute the general case for a N x N matrix where N >= 2
    match determinant::<T>(matrix) {
        Some(det) => {
            if det == T::zero() {
                return None;
            }
            let determinant_reciprocal = T::one() / det;
            let mut cofactor_matrix = Matrix::empty(T::zero(), matrix.size());
            for i in 0..matrix.rows() {
                for j in 0..matrix.columns() {
                    // this should always return Some due to the earlier checks
                    let ij_minor = minor::<T>(matrix, i, j)?;
                    // i and j may each be up to the maximum value for usize but
                    // we only need to know if they are even or odd as
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
            // finally to compute the inverse we need to multiply each element by 1 / |A|
            cofactor_matrix.map_mut(|element| element * determinant_reciprocal.clone());
            Some(cofactor_matrix)
        }
        None => None,
    }
}

/**
 * Computes the inverse of a matrix provided that it exists. To have an inverse
 * a matrix must be square (same number of rows and columns) and it must also
 * have a non zero determinant.
 *
 * The first dimension in the Tensor's shape will be taken as the rows of the matrix, and the
 * second dimension as the columns. If you instead have columns and then rows for the Tensor's
 * shape, you should reorder the Tensor before calling this function to get the appropriate
 * matrix.
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
 * `linear_algebra::inverse:_tensor:<f32>(&tensor)`
 *
 * Alternatively, the compiler doesn't seem to run into this problem if you
 * use the equivalent methods on the tensor type like so:
 * `tensor.inverse()`
 */
pub fn inverse_tensor<T, S, I>(tensor: I) -> Option<Tensor<T, 2>>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    I: Into<TensorView<T, S, 2>>,
    S: TensorRef<T, 2>,
{
    inverse_less_generic::<T, S>(tensor.into())
}

fn inverse_less_generic<T, S>(tensor: TensorView<T, S, 2>) -> Option<Tensor<T, 2>>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S: TensorRef<T, 2>,
{
    let shape = tensor.shape();
    if !crate::tensors::dimensions::is_square(&shape) {
        return None;
    }

    // inverse of a 1 x 1 matrix is a special case
    if shape[0].1 == 1 {
        // determinant of a 1 x 1 matrix is the single element
        let element = tensor
            .iter()
            .next()
            .expect("1x1 tensor must have a single element");
        if element == T::zero() {
            return None;
        }
        return Some(Tensor::from(shape, vec![T::one() / element]));
    }

    // compute the general case for a N x N matrix where N >= 2
    match determinant_less_generic::<T, _>(&tensor) {
        Some(det) => {
            if det == T::zero() {
                return None;
            }
            let determinant_reciprocal = T::one() / det;
            let mut cofactor_matrix = Tensor::empty(shape, T::zero());
            for ([i, j], x) in cofactor_matrix.iter_reference_mut().with_index() {
                // this should always return Some due to the earlier checks
                let ij_minor = minor_tensor::<T, _>(&tensor, i, j)?;
                // i and j may each be up to the maximum value for usize but
                // we only need to know if they are even or odd as
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
                *x = sign * ij_minor;
            }
            // tranposing the cofactor matrix yields the adjugate matrix
            cofactor_matrix.transpose_mut([shape[1].0, shape[0].0]);
            // finally to compute the inverse we need to multiply each element by 1 / |A|
            cofactor_matrix.map_mut(|element| element * determinant_reciprocal.clone());
            Some(cofactor_matrix)
        }
        None => None,
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
where
    for<'a> &'a T: NumericRef<T>,
{
    minor_mut::<T>(&mut matrix.clone(), i, j)
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
where
    for<'a> &'a T: NumericRef<T>,
{
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
    determinant::<T>(matrix)
}

fn minor_tensor<T, S>(tensor: &TensorView<T, S, 2>, i: usize, j: usize) -> Option<T>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S: TensorRef<T, 2>,
{
    use crate::tensors::views::{IndexRange, TensorMask};
    let shape = tensor.shape();
    if shape[0].1 == 1 || shape[1].1 == 1 {
        // nothing to delete
        return None;
    }
    if !crate::tensors::dimensions::is_square(&shape) {
        // no determinant
        return None;
    }
    let minored = TensorView::from(
        TensorMask::from_all(
            tensor.source_ref(),
            [Some(IndexRange::new(i, 1)), Some(IndexRange::new(j, 1))],
        )
        .expect("Having just checked tensor is at least 2x2 we should be able to take a mask"),
    );
    determinant_less_generic::<T, _>(&minored)
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
 * [`matrix.determinant()`](Matrix::determinant)
 */
pub fn determinant<T: Numeric>(matrix: &Matrix<T>) -> Option<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let length = matrix.rows();

    match length {
        0 => return None,
        1 => return Some(matrix.scalar()),
        _ => (),
    };

    determinant_less_generic::<T, _>(&TensorView::from(
        crate::interop::TensorRefMatrix::from(matrix).ok()?,
    ))
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
 * The first dimension in the Tensor's shape will be taken as the rows of the matrix, and the
 * second dimension as the columns. If you instead have columns and then rows for the Tensor's
 * shape, you should reorder the Tensor before calling this function to get the appropriate
 * matrix.
 *
 * Note that the determinant of a 1 x 1 matrix is just the element in the matrix.
 *
 * This function computes the determinant using the same type as that of the Tensor,
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
 * `linear_algebra::determinant_tensor::<f32, _, _>(&tensor)`
 *
 * Alternatively, the compiler doesn't seem to run into this problem if you
 * use the equivalent methods on the tensor type like so:
 * [`tensor.determinant()`](Tensor::determinant)
 */
pub fn determinant_tensor<T, S, I>(tensor: I) -> Option<T>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    I: Into<TensorView<T, S, 2>>,
    S: TensorRef<T, 2>,
{
    determinant_less_generic::<T, S>(&tensor.into())
}

fn determinant_less_generic<T, S>(tensor: &TensorView<T, S, 2>) -> Option<T>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S: TensorRef<T, 2>,
{
    let shape = tensor.shape();
    if !crate::tensors::dimensions::is_square(&shape) {
        return None;
    }
    let length = shape[0].1;

    if length == 0 {
        return None;
    }

    let matrix = tensor.index();

    if length == 1 {
        return Some(matrix.get([0, 0]));
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
            let element = matrix.get_reference([n, *i]);
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
where
    F: FnMut(&mut Vec<T>),
{
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
 * In place version of generate_permutations which calls the consumer on
 * each permuted list without performing any copies (ie each permuted list)
 * is the same list before and after permutation.
 */
fn with_each_permutation<T: Clone, F>(list: &mut Vec<T>, consumer: &mut F)
where
    F: FnMut(&mut Vec<T>, bool),
{
    let mut even_swaps = true;
    heaps_permutations(list.len(), list, &mut |permuted| {
        consumer(permuted, even_swaps);
        even_swaps = !even_swaps;
    });
}

#[cfg(test)]
#[test]
fn test_permutations() {
    // Exhaustively test permutation even/oddness for an input
    // of length 3
    let mut list = vec![1, 2, 3];
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
    let mut list = vec![1, 2, 3, 4, 5];
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
 * The covariance matrix is a matrix of how each feature varies with itself
 * (along the diagonal) and all the other features (symmetrically above and below
 * the diagonal).
 *
 * Each element in the covariance matrix at (i, j) will be the variance of the
 * ith and jth features from the feature matrix, defined as the zero meaned
 * dot product of the two feature vectors divided by the number of samples.
 *
 * If all the features in the input have a variance of one then the covariance matrix
 * returned by this function will be equivalent to the correlation matrix of the input
 *
 * This function does not perform [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction)
 *
 * # Panics
 *
 * If the numeric type is unable to represent the number of samples
 * for each feature (ie if `T: i8` and you have 1000 samples) then this function
 * will panic.
 *
 * # Warning
 *
 * With some uses of this function the Rust compiler gets confused about what type `T`
 * should be and you will get the error:
 * > overflow evaluating the requirement `&'a _: easy_ml::numeric::NumericByValue<_, _>`
 *
 * In this case you need to manually specify the type of T by using the
 * turbofish syntax like:
 * `linear_algebra::covariance_column_features::<f32>(&matrix)`
 *
 * Alternatively, the compiler doesn't seem to run into this problem if you
 * use the equivalent methods on the matrix type like so:
 * `matrix.covariance_column_features()`
 */
pub fn covariance_column_features<T: Numeric>(matrix: &Matrix<T>) -> Matrix<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    let features = matrix.columns();
    let samples = T::from_usize(matrix.rows())
        .expect("The maximum value of the matrix type T cannot represent this many samples");
    let mut covariance_matrix = Matrix::empty(T::zero(), (features, features));
    covariance_matrix.map_mut_with_index(|_, i, j| {
        // set each element of the covariance matrix to the variance
        // of features i and j
        let feature_i_mean: T = matrix.column_iter(i).sum::<T>() / &samples;
        let feature_j_mean: T = matrix.column_iter(j).sum::<T>() / &samples;
        matrix
            .column_reference_iter(i)
            .map(|x| x - &feature_i_mean)
            .zip(matrix.column_reference_iter(j).map(|y| y - &feature_j_mean))
            .map(|(x, y)| x * y)
            .sum::<T>()
            / &samples
    });
    covariance_matrix
}

/**
 * Computes the covariance matrix for an NxM feature matrix, in which
 * each M'th column has N features to find the covariance and variance of.
 *
 * The covariance matrix is a matrix of how each feature varies with itself
 * (along the diagonal) and all the other features (symmetrically above and below
 * the diagonal).
 *
 * Each element in the covariance matrix at (i, j) will be the variance of the
 * ith and jth features from the feature matrix, defined as the zero meaned
 * dot product of the two feature vectors divided by the number of samples.
 *
 * If all the features in the input have a variance of one then the covariance matrix
 * returned by this function will be equivalent to the correlation matrix of the input
 *
 * This function does not perform [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction)
 *
 * # Panics
 *
 * If the numeric type is unable to represent the number of samples
 * for each feature (ie if `T: i8` and you have 1000 samples) then this function
 * will panic.
 *
 * # Warning
 *
 * With some uses of this function the Rust compiler gets confused about what type `T`
 * should be and you will get the error:
 * > overflow evaluating the requirement `&'a _: easy_ml::numeric::NumericByValue<_, _>`
 *
 * In this case you need to manually specify the type of T by using the
 * turbofish syntax like:
 * `linear_algebra::covariance_row_features::<f32>(&matrix)`
 *
 * Alternatively, the compiler doesn't seem to run into this problem if you
 * use the equivalent methods on the matrix type like so:
 * `matrix.covariance_row_features()`
 */
pub fn covariance_row_features<T: Numeric>(matrix: &Matrix<T>) -> Matrix<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    let features = matrix.rows();
    let samples = T::from_usize(matrix.columns())
        .expect("The maximum value of the matrix type T cannot represent this many samples");
    let mut covariance_matrix = Matrix::empty(T::zero(), (features, features));
    covariance_matrix.map_mut_with_index(|_, i, j| {
        // set each element of the covariance matrix to the variance
        // of features i and j
        let feature_i_mean: T = matrix.row_iter(i).sum::<T>() / &samples;
        let feature_j_mean: T = matrix.row_iter(j).sum::<T>() / &samples;
        matrix
            .row_reference_iter(i)
            .map(|x| x - &feature_i_mean)
            .zip(matrix.row_reference_iter(j).map(|y| y - &feature_j_mean))
            .map(|(x, y)| x * y)
            .sum::<T>()
            / &samples
    });
    covariance_matrix
}

/**
 * Computes the covariance matrix for a 2 dimensional Tensor feature matrix.
 *
 * The `feature_dimension` specifies which dimension holds the features. The other dimension
 * is assumed to hold the samples. For a Tensor with a `feature_dimension` of length N, and
 * the other dimension of length M, returns an NxN covariance matrix with a shape of
 * `[("i", N), ("j", N)]`.
 *
 * Each element in the covariance matrix at (i, j) will be the variance of the ith and jth
 * features from the feature matrix, defined as the zero meaned dot product of the two
 * feature vectors divided by the number of samples (M).
 *
 * If all the features in the input have a variance of one then the covariance matrix
 * returned by this function will be equivalent to the correlation matrix of the input
 *
 * This function does not perform [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction)
 *
 * # Panics
 *
 * - If the numeric type is unable to represent the number of samples for each feature
 * (ie if `T: i8` and you have 1000 samples)
 * - If the provided feature_dimension is not a dimension in the tensor
 *
 * # Warning
 *
 * With some uses of this function the Rust compiler gets confused about what type `T`
 * should be and you will get the error:
 * > overflow evaluating the requirement `&'a _: easy_ml::numeric::NumericByValue<_, _>`
 *
 * In this case you need to manually specify the type of T by using the
 * turbofish syntax like:
 * `linear_algebra::covariance::<f32, _, _>(&matrix)`
 *
 * Alternatively, the compiler doesn't seem to run into this problem if you
 * use the equivalent methods on the matrix type like so:
 * `tensor.covariance("features")`
 *
 * # Example
 *
 * ```
 * use easy_ml::tensors::Tensor;
 * let matrix = Tensor::from([("samples", 5), ("features", 3)], vec![
 * //  X     Y    Z
 *     1.0,  0.0, 0.5,
 *     1.2, -1.0, 0.4,
 *     1.8, -1.2, 0.7,
 *     0.9,  0.1, 0.3,
 *     0.7,  0.5, 0.6
 * ]);
 * let covariance_matrix = matrix.covariance("features");
 * let (x, y, z) = (0, 1, 2);
 * let x_y_z = covariance_matrix.index();
 * // the variance of each feature with itself is positive
 * assert!(x_y_z.get([x, x]) > 0.0);
 * assert!(x_y_z.get([y, y]) > 0.0);
 * assert!(x_y_z.get([z, z]) > 0.0);
 * // first feature X and second feature Y have negative covariance (as X goes up Y goes down)
 * assert!(x_y_z.get([x, y]) < 0.0);
 * println!("{}", covariance_matrix);
 * // D = 2
 * // ("i", 3), ("j", 3)
 * // [ 0.142, -0.226, 0.026
 * //   -0.226, 0.438, -0.022
 * //   0.026, -0.022, 0.020 ]
 * ```
 */
#[track_caller]
pub fn covariance<T, S, I>(tensor: I, feature_dimension: Dimension) -> Tensor<T, 2>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    I: Into<TensorView<T, S, 2>>,
    S: TensorRef<T, 2>,
{
    covariance_less_generic::<T, S>(tensor.into(), feature_dimension)
}

#[track_caller]
fn covariance_less_generic<T, S>(
    tensor: TensorView<T, S, 2>,
    feature_dimension: Dimension,
) -> Tensor<T, 2>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
    S: TensorRef<T, 2>,
{
    let shape = tensor.shape();
    let features_index = {
        if shape[0].0 == feature_dimension {
            0
        } else if shape[1].0 == feature_dimension {
            1
        } else {
            panic!(
                "Feature dimension {:?} is not present in the input tensor's shape: {:?}",
                feature_dimension, shape
            );
        }
    };
    let (feature_dimension, features) = shape[features_index];
    let (_sample_dimension, samples) = shape[1 - features_index];
    let samples = T::from_usize(samples)
        .expect("The maximum value of the matrix type T cannot represent this many samples");
    let mut covariance_matrix = Tensor::empty([("i", features), ("j", features)], T::zero());
    covariance_matrix.map_mut_with_index(|[i, j], _| {
        // set each element of the covariance matrix to the variance of features i and j
        #[rustfmt::skip]
        let feature_i_mean: T = tensor
            .select([(feature_dimension, i)])
            .iter()
            .sum::<T>() / &samples;
        #[rustfmt::skip]
        let feature_j_mean: T = tensor
            .select([(feature_dimension, j)])
            .iter()
            .sum::<T>() / &samples;
        tensor
            .select([(feature_dimension, i)])
            .iter_reference()
            .map(|x| x - &feature_i_mean)
            .zip(
                tensor
                    .select([(feature_dimension, j)])
                    .iter_reference()
                    .map(|y| y - &feature_j_mean),
            )
            .map(|(x, y)| x * y)
            .sum::<T>()
            / &samples
    });
    covariance_matrix
}

/**
 * Computes the mean of the values in an iterator, consuming the iterator.
 *
 * This function does not perform [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction)
 *
 * # Panics
 *
 * If the iterator is empty. This function will also fail if the length of the iterator
 * or sum of all the values in the iterator exceeds the maximum number the type can
 * represent.
 */
pub fn mean<I, T: Numeric>(mut data: I) -> T
where
    I: Iterator<Item = T>,
{
    let mut next = data.next();
    assert!(next.is_some(), "Provided iterator must not be empty");
    let mut count = T::zero();
    let mut sum = T::zero();
    while next.is_some() {
        count = count + T::one();
        sum = sum + next.unwrap();
        next = data.next();
    }
    sum / count
}

/**
 * Computes the variance of the values in an iterator, consuming the iterator.
 *
 * Variance is defined as expected value of of the squares of the zero mean data.
 * It captures how much data varies from its mean, ie the spread of the data.
 *
 * This function does not perform [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction)
 *
 * Variance may also be computed as the mean of each squared datapoint minus the
 * square of the mean of the data. Although this method would allow for a streaming
 * implementation the [wikipedia page](https://en.wikipedia.org/wiki/Variance#Definition)
 * cautions: "This equation should not be used for computations using floating point
 * arithmetic because it suffers from catastrophic cancellation if the two components
 * of the equation are similar in magnitude".
 *
 * # Panics
 *
 * If the iterator is empty. This function will also fail if the length of the iterator
 * or sum of all the values in the iterator exceeds the maximum number the type can
 * represent.
 */
pub fn variance<I, T: Numeric>(data: I) -> T
where
    I: Iterator<Item = T>,
{
    let mut list = data.collect::<Vec<T>>();
    assert!(!list.is_empty(), "Provided iterator must not be empty");

    // copy the list as we need to keep it as well as getting the mean
    let m = mean(list.iter().cloned());
    // use drain here as we no longer need to keep list
    mean(
        list.drain(..)
            .map(|x| (x.clone() - m.clone()) * (x - m.clone())),
    )
}

/**
 * Computes the softmax of the values in an iterator, consuming the iterator.
 *
 * softmax(z)\[i\] = e<sup>z<sub>i</sub></sup> / the sum of e<sup>z<sub>j</sub></sup> for all j
 * where z is a list of elements
 *
 * Softmax normalises an input of numbers into a probability distribution, such
 * that they will sum to 1. This is often used to make a neural network
 * output a single number.
 *
 * The implementation shifts the inputs by the maximum value in the iterator,
 * to make numerical overflow less of a problem. As per the definition of softmax,
 * softmax(z) = softmax(z-max(z)).
 *
 * [Further information](https://en.wikipedia.org/wiki/Softmax_function)
 *
 * # Panics
 *
 * If the iterator contains NaN values, or any value for which PartialOrd fails.
 *
 * This function will also fail if the length of the iterator or sum of all the values
 * in the iterator exceeds the maximum or minimum number the type can represent.
 */
pub fn softmax<I, T: Numeric + Real>(data: I) -> Vec<T>
where
    I: Iterator<Item = T>,
{
    let list = data.collect::<Vec<T>>();
    if list.is_empty() {
        return Vec::with_capacity(0);
    }
    let max = list
        .iter()
        .max_by(|a, b| a.partial_cmp(b).expect("NaN should not be in list"))
        .unwrap();

    let denominator: T = list.iter().cloned().map(|x| (x - max).exp()).sum();
    list.iter()
        .cloned()
        .map(|x| (x - max).exp() / denominator.clone())
        .collect()
}

/**
 * Computes the F-1 score of the Precision and Recall
 *
 * 2 * (precision * recall) / (precision + recall)
 *
 * # [F-1 score](https://en.wikipedia.org/wiki/F1_score)
 *
 * This is a harmonic mean of the two, which penalises the score
 * more heavily if either the precision or recall are poor than
 * an arithmetic mean.
 *
 * The F-1 score is a helpful metric for assessing classifiers, as
 * it takes into account that classes may be heavily biased which
 * Accuracy does not. For example, it may be quite easy to create a
 * 95% accurate test for a medical condition, which inuitively seems
 * very good, but if 99.9% of patients are expected to not have the
 * condition then accuracy is a poor way to measure performance because
 * it does not consider that the cost of false negatives is very high.
 *
 * Note that Precision and Recall both depend on there being a positive
 * and negative class for a classification task, in some contexts this may
 * be an arbitrary choice.
 *
 * # [Precision](https://en.wikipedia.org/wiki/Precision_and_recall)
 *
 * In classification, precision is true positives / positive predictions.
 * It measures correct identifications of the positive class compared
 * to all predictions of the positive class. You can trivially get
 * 100% precision by never predicting the positive class, as this can
 * never result in a false positive.
 *
 * Note that the meaning of precision in classification or document
 * retrieval is not the same as its meaning in [measurements](https://en.wikipedia.org/wiki/Accuracy_and_precision).
 *
 * # [Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
 *
 * In classification, recall is true positives / actual positives.
 * It measures how many of the positive cases are identified. You
 * can trivially get 100% recall by always predicting the positive class,
 * as this can never result in a false negative.
 *
 * [F scores](https://en.wikipedia.org/wiki/F1_score)
 *
 * The F-1 score is an evenly weighted combination of Precision and
 * Recall. For domains where the cost of false positives and false
 * negatives are not equal, you should use a biased F score that weights
 * precision or recall more strongly than the other.
 */
pub fn f1_score<T: Numeric>(precision: T, recall: T) -> T {
    (T::one() + T::one()) * ((precision.clone() * recall.clone()) / (precision + recall))
}

/**
 * Computes the cholesky decomposition of a matrix. This yields a matrix `L`
 * such that for the provided matrix `A`, `L * L^T = A`. `L` will always be
 * lower triangular, ie all entries above the diagonal will be 0. Hence cholesky
 * decomposition can be interpreted as a generalised square root function.
 *
 * Cholesky decomposition is defined for
 * [symmetric](https://en.wikipedia.org/wiki/Hermitian_matrix),
 * [positive definite](https://en.wikipedia.org/wiki/Definite_matrix) matrices.
 *
 * This function does not check that the provided matrix is symmetric. However, given a symmetric
 * input, if the input is not positive definite `None` is returned. Attempting a cholseky
 * decomposition is also an efficient way to check if such a matrix is positive definite.
 * In the future additional checks that the input is valid could be added.
 */
pub fn cholesky_decomposition<T: Numeric + Sqrt<Output = T>>(
    matrix: &Matrix<T>,
) -> Option<Matrix<T>>
where
    for<'a> &'a T: NumericRef<T>,
{
    if matrix.rows() != matrix.columns() {
        return None;
    }
    // The computation steps are outlined nicely at https://rosettacode.org/wiki/Cholesky_decomposition
    let mut lower_triangular = Matrix::empty(T::zero(), matrix.size());
    let n = lower_triangular.rows();
    for i in 0..n {
        // For each column j we need to compute all i, j entries
        // before incrementing j further as the diagonals depend
        // on the elements below the diagonal of the previous columns,
        // and the elements below the diagonal depend on the diagonal
        // of their column and elements below the diagonal up to that
        // column.
        for j in 0..=i {
            // For the i = j case we compute the sum of squares, otherwise we're
            // computing a sum of L_ik * L_jk using the current column and prior columns
            let sum = {
                let mut sum = T::zero();
                for k in 0..j {
                    sum = &sum
                        + (lower_triangular.get_reference(i, k)
                            * lower_triangular.get_reference(j, k));
                }
                sum
            };
            // Calculate L_ij as we step through the lower diagonal
            lower_triangular.set(
                i,
                j,
                if i == j {
                    let entry_squared = matrix.get_reference(i, j) - sum;
                    if entry_squared <= T::zero() {
                        // input wasn't positive definite! avoid sqrt of a negative number.
                        // We can take sqrt(0) but that will leave a 0 on the diagonal which
                        // will then cause division by zero for the j < i case later.
                        return None;
                    }
                    entry_squared.sqrt()
                } else /* j < i */ {
                    (matrix.get_reference(i, j) - sum) *
                        (T::one() / lower_triangular.get_reference(j, j))
                }
            );
        }
    }
    Some(lower_triangular)
}

/**
 * Computes the cholesky decomposition of a Tensor matrix. This yields a matrix `L`
 * such that for the provided matrix `A`, `L * L^T = A`. `L` will always be
 * lower triangular, ie all entries above the diagonal will be 0. Hence cholesky
 * decomposition can be interpreted as a generalised square root function.
 *
 * Cholesky decomposition is defined for
 * [symmetric](https://en.wikipedia.org/wiki/Hermitian_matrix),
 * [positive definite](https://en.wikipedia.org/wiki/Definite_matrix) matrices.
 *
 * This function does not check that the provided matrix is symmetric. However, given a symmetric
 * input, if the input is not positive definite `None` is returned. Attempting a cholseky
 * decomposition is also an efficient way to check if such a matrix is positive definite.
 * In the future additional checks that the input is valid could be added.
 *
 * The output matrix will have the same shape as the input.
 */
pub fn cholesky_decomposition_tensor<T, S, I>(tensor: I) -> Option<Tensor<T, 2>>
where
    T: Numeric + Sqrt<Output = T>,
    for<'a> &'a T: NumericRef<T>,
    I: Into<TensorView<T, S, 2>>,
    S: TensorRef<T, 2>,
{
    cholesky_decomposition_less_generic::<T, S>(&tensor.into())
}

fn cholesky_decomposition_less_generic<T, S>(tensor: &TensorView<T, S, 2>) -> Option<Tensor<T, 2>>
where
    T: Numeric + Sqrt<Output = T>,
    for<'a> &'a T: NumericRef<T>,
    S: TensorRef<T, 2>,
{
    // TODO: Port matrix implementation and delegate matrix API to the tensor one to avoid copies
    let shape = tensor.shape();
    let matrix = Matrix::from_flat_row_major((shape[0].1, shape[1].1), tensor.iter().collect());
    let lower_triangular = cholesky_decomposition::<T>(&matrix)?;
    (lower_triangular, [shape[0].0, shape[1].0]).try_into().ok()
}

/**
 * The result of an `LDL^T` Decomposition of some matrix `A` such that `LDL^T = A`.
 */
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LDLTDecomposition<T> {
    pub l: Matrix<T>,
    pub d: Matrix<T>,
}

impl<T: std::fmt::Display + Clone> std::fmt::Display for LDLTDecomposition<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "L:\n{}", &self.l)?;
        write!(f, "D:\n{}", &self.d)
    }
}

impl<T> LDLTDecomposition<T> {
    /**
     * Creates an `LDL^T` Decomposition struct from two matrices without checking that `LDL^T = A`
     * or that L and D have the intended properties.
     *
     * This is provided for assistance with unit testing.
     */
    pub fn from_unchecked(l: Matrix<T>, d: Matrix<T>) -> LDLTDecomposition<T> {
        LDLTDecomposition { l, d }
    }
}

/**
 * Computes the LDL^T decomposition of a matrix. This yields a matrix `L` and a matrix `D`
 * such that for the provided matrix `A`, `L * D * L^T = A`. `L` will always be
 * unit lower triangular, ie all entries above the diagonal will be 0, and all entries along
 * the diagonal will br 1. `D` will always contain zeros except along the diagonal. This
 * decomposition is closely related to the [cholesky decomposition](cholesky_decomposition)
 * with the notable difference that it avoids taking square roots.
 *
 * Similarly to the cholseky decomposition, the input matrix must be
 * [symmetric](https://en.wikipedia.org/wiki/Hermitian_matrix) and
 * [positive definite](https://en.wikipedia.org/wiki/Definite_matrix).
 *
 * This function does not check that the provided matrix is symmetric. However, given a symmetric
 * input, if the input is only positive **semi**definite `None` is returned. In the future
 * additional checks that the input is valid could be added.
 *
 * # Warning
 *
 * With some uses of this function the Rust compiler gets confused about what type `T`
 * should be and you will get the error:
 * > overflow evaluating the requirement `&'a _: easy_ml::numeric::NumericByValue<_, _>`
 *
 * In this case you need to manually specify the type of T by using the
 * turbofish syntax like:
 * `linear_algebra::ldlt_decomposition::<f32>(&matrix)`
 */
pub fn ldlt_decomposition<T>(
    matrix: &Matrix<T>,
) -> Option<LDLTDecomposition<T>>
where
    T: Numeric,
    for<'a> &'a T: NumericRef<T>,
{
    // The algorithm is outlined nicely in context as Algorithm 1.2 here:
    // https://mcsweeney90.github.io/files/modified-cholesky-decomposition-and-applications.pdf
    // and also as proper code here (though a less efficient solution):
    // https://astroanddata.blogspot.com/2020/04/ldl-decomposition-with-python.html
    if matrix.rows() != matrix.columns() {
        return None;
    }
    let mut lower_triangular = Matrix::empty(T::zero(), matrix.size());
    let mut diagonal = Matrix::empty(T::zero(), matrix.size());
    let n = lower_triangular.rows();
    for j in 0..n {
        let sum = {
            let mut sum = T::zero();
            for k in 0..j {
                sum = &sum + (
                    lower_triangular.get_reference(j, k) * lower_triangular.get_reference(j, k) *
                        diagonal.get_reference(k, k)
                );
            }
            sum
        };
        diagonal.set(
            j,
            j,
            {
                let entry = matrix.get_reference(j, j) - sum;
                if entry == T::zero() {
                    // If input is positive definite then no diagonal will be 0. Otherwise we
                    // fail the decomposition to avoid division by zero in the j < i case later.
                    // Note: unlike cholseky, negatives here are fine since we can still perform
                    // the calculations sensibly.
                    return None;
                }
                entry
            }
        );
        for i in j..n {
            lower_triangular.set(
                i,
                j,
                if i == j {
                    T::one()
                } else /* j < i */ {
                    let sum = {
                        let mut sum = T::zero();
                        for k in 0..j {
                            sum = &sum + (
                                lower_triangular.get_reference(i, k) * lower_triangular.get_reference(j, k) *
                                    diagonal.get_reference(k, k)
                            );
                        }
                        sum
                    };
                    (matrix.get_reference(i, j) - sum) * (T::one() / diagonal.get_reference(j, j))
                }
            )
        }
    }
    Some(LDLTDecomposition {
        l: lower_triangular,
        d: diagonal,
    })
}

/**
 * The result of a QR Decomposition of some matrix A such that `QR = A`.
 */
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct QRDecomposition<T> {
    pub q: Matrix<T>,
    pub r: Matrix<T>,
}

impl<T: std::fmt::Display + Clone> std::fmt::Display for QRDecomposition<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "Q:\n{}", &self.q)?;
        write!(f, "R:\n{}", &self.r)
    }
}

impl<T> QRDecomposition<T> {
    /**
     * Creates a QR Decomposition struct from two matrices without checking that `QR = A`
     * or that Q and R have the intended properties.
     *
     * This is provided for assistance with unit testing.
     */
    pub fn from_unchecked(q: Matrix<T>, r: Matrix<T>) -> QRDecomposition<T> {
        QRDecomposition { q, r }
    }
}

/**
 * Computes the householder matrix along the column vector input.
 *
 * For an Mx1 input, the householder matrix output will be MxM
 */
fn householder_matrix<T: Numeric + Real>(matrix: Matrix<T>) -> Matrix<T>
where
    for<'a> &'a T: NumericRef<T> + RealRef<T>,
{
    // The computation steps are outlined nicely at https://en.wikipedia.org/wiki/QR_decomposition#Using_Householder_reflections
    // Supporting reference implementations are on Rosettacode https://rosettacode.org/wiki/QR_decomposition
    // we hardcode to taking the first column vector of the input matrix
    assert_eq!(matrix.columns(), 1, "Input must be a column vector");
    let x = matrix;
    let rows = x.rows();
    let length = x.euclidean_length();
    let a = {
        // we hardcode to wanting to zero all elements below the first
        let sign = x.get(0, 0);
        if sign > T::zero() {
            length
        } else {
            -length
        }
    };
    let u = {
        // u = x - ae, where e is [1 0 0 0 ... 0]^T, and x is the column vector so
        // u is equal to x except for the first element.
        // Also, we invert the sign of a to avoid loss of significance, so u[0] becomes x[0] + a
        let mut u = x;
        u.set(0, 0, u.get(0, 0) + a);
        u
    };
    // v = u / ||u||
    let v = {
        let length = u.euclidean_length();
        u / length
    };
    let identity = Matrix::diagonal(T::one(), (rows, rows));
    let two = T::one() + T::one();
    // I - 2 v v^T
    identity - ((&v * v.transpose()) * two)
}

/**
 * Computes a QR decomposition of a MxN matrix where M >= N.
 *
 * For an input matrix A, decomposes this matrix into a product of QR, where Q is an
 * [orthogonal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix) and R is an
 * upper triangular matrix (all entries below the diagonal are 0), and QR = A.
 *
 * If the input matrix has more columns than rows, returns None.
 *
 * # Warning
 *
 * With some uses of this function the Rust compiler gets confused about what type `T`
 * should be and you will get the error:
 * > overflow evaluating the requirement `&'a _: easy_ml::numeric::NumericByValue<_, _>`
 *
 * In this case you need to manually specify the type of T by using the
 * turbofish syntax like:
 * `linear_algebra::qr_decomposition::<f32>(&matrix)`
 */
pub fn qr_decomposition<T: Numeric + Real>(matrix: &Matrix<T>) -> Option<QRDecomposition<T>>
where
    for<'a> &'a T: NumericRef<T> + RealRef<T>,
{
    if matrix.columns() > matrix.rows() {
        return None;
    }
    // The computation steps are outlined nicely at https://en.wikipedia.org/wiki/QR_decomposition#Using_Householder_reflections
    // Supporting reference implementations are at Rosettacode https://rosettacode.org/wiki/QR_decomposition
    let iterations = std::cmp::min(matrix.rows() - 1, matrix.columns());
    let mut q = None;
    let mut r = matrix.clone();
    for column in 0..iterations {
        // Conceptually, on each iteration we take a minor of r to retain the bottom right of
        // the matrix, with one fewer row/column on each iteration since that will have already
        // been zeroed. However, we then immediately discard all but the first column of that
        // minor, so we skip the minor step and compute directly the first column of the minor
        // we would have taken.
        // let submatrix = r.retain(
        //     Slice2D::new()
        //         .rows(Slice::Range(column..matrix.rows()))
        //         .columns(Slice::Range(column..matrix.columns()))
        // );
        // let submatrix_first_column = Matrix::column(submatrix.column_iter(0).collect());
        let submatrix_first_column = Matrix::column(r.column_iter(column).skip(column).collect());
        // compute the (M-column)x(M-column) householder matrix
        let h = householder_matrix::<T>(submatrix_first_column);
        // pad the h into the bottom right of an identity matrix so it is MxM
        // like so:
        // 1 0 0
        // 0 H H
        // 0 H H
        let h = {
            let mut identity = Matrix::diagonal(T::one(), (matrix.rows(), matrix.rows()));
            for i in 0..h.rows() {
                for j in 0..h.columns() {
                    identity.set(column + i, column + j, h.get(i, j));
                }
            }
            identity
        };
        // R = H_n * ... H_3 * H_2 * H_1 * A
        r = &h * r;
        // Q = H_1 * H_2 * H_3 .. H_n
        match q {
            None => q = Some(h),
            Some(h_previous) => q = Some(h_previous * h),
        }
    }
    Some(QRDecomposition {
        // This should always be Some because the input matrix has to be at least 1x1
        q: q.unwrap(),
        r,
    })
}
