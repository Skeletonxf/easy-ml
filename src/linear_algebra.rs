use crate::matrices::{Matrix, Row, Column};
use crate::numeric::Numeric;

/**
 * Computes the inverse of a matrix
 * https://en.wikipedia.org/wiki/Invertible_matrix#Analytic_solution
 */
pub fn inverse<T: Numeric>(matrix: &Matrix<T>) -> Option<Matrix<T>> {
    // TODO
    None
}

/**
 * Computes the determinant of a matrix
 * https://en.wikipedia.org/wiki/Determinant#n_%C3%97_n_matrices
 */
pub fn determinant<T: Numeric>(matrix: &Matrix<T>) -> Option<T> {
    if matrix.columns() != matrix.rows() {
        return None;
    }
    let length = matrix.columns();

    // TODO: need to check that these permutations are even and odd successively
    // for use in computing the signature of the each permutation set
    let all_permutations = permutations((1..=length).collect());
    let sum = 0; // TODO need trait to define zero value for the type

    // TODO
    None
}

/*
 * Computes the factorial of a number.
 * eg for an input of 5 computes 1 * 2 * 3 * 4 * 5
 * which is equal to 120
 */
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/*
 * Recursive computation of all permutations of a list
 * the standard library method isn't stable yet
 * https://doc.rust-lang.org/1.1.0/collections/slice/struct.Permutations.html
 */
fn permutations<T: Clone>(list: Vec<T>) -> Vec<Vec<T>> {
    if list.is_empty() {
        return vec![];
    }
    if list.len() == 1 {
        return vec![list];
    }
    let mut all_permutations: Vec<Vec<T>> = Vec::with_capacity(factorial(list.len()));
    for n in 0..list.len() {
        let mut list = list.clone();
        let element = list.remove(n);
        // get each permutation of the remaining elements
        // and add the element we removed to the permuted list
        for mut sub_permutation in permutations(list).drain(..) {
            sub_permutation.push(element.clone());
            all_permutations.push(sub_permutation);
        }
    }
    all_permutations
}
