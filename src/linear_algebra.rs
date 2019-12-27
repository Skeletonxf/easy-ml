/*!
 * Linear algebra algorithms on numbers and matrices.
 *
 * Note that these functions are also exposed as corresponding methods on the Matrix type,
 * but in depth documentation is only presented here.
 */

use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;

use crate::matrices::{Matrix};
use crate::numeric::Numeric;

// /**
//  * Computes the inverse of a matrix
//  * https://en.wikipedia.org/wiki/Invertible_matrix#Analytic_solution
//  */
// fn inverse<T: Numeric>(matrix: &Matrix<T>) -> Option<Matrix<T>> {
//     // TODO
//     None
// }

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
 * This function computes the determinant using the same type as that of the Matrix,
 * hence if the input type is unsigned (such as Wrapping&lt;u8&gt;) the value computed
 * is likely to not make any sense because a determinant may be negative.
 *
 * [https://en.wikipedia.org/wiki/Determinant](https://en.wikipedia.org/wiki/Determinant)
 */
pub fn determinant<T: Numeric>(matrix: &Matrix<T>) -> Option<T>
where T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> {
    if matrix.columns() != matrix.rows() {
        return None;
    }
    let length = matrix.columns();

    if length == 0 {
        return None;
    }

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
            let element = matrix.get(n, *i);
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
