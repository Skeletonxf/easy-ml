extern crate easy_ml;

#[cfg(test)]
mod tests {
    use easy_ml::matrices::Matrix;
    use easy_ml::linear_algebra;

    #[test]
    fn check_determinant_2_by_2() {
        let matrix = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let a = matrix.get(0, 0);
        let b = matrix.get(0, 1);
        let c = matrix.get(1, 0);
        let d = matrix.get(1, 1);
        let determinant = (a * d) - (b * c);
        // FIXME
        assert_eq!(determinant, linear_algebra::determinant(&matrix).unwrap());
    }

    #[test]
    fn check_determinant_3_by_3() {
        // use the example from wikipedia
        // https://en.wikipedia.org/wiki/Determinant#n_%C3%97_n_matrices
        let matrix = Matrix::from(vec![
            vec![-1.0,  2.0,  3.0],
            vec![ 3.0,  4.0, -5.0],
            vec![ 5.0, -2.0,  3.0]]);

        let determinant = 0.0
            + (matrix.get(0, 0) * matrix.get(1, 1) * matrix.get(2, 2))
            - (matrix.get(0, 0) * matrix.get(1, 2) * matrix.get(2, 1))
            - (matrix.get(0, 1) * matrix.get(1, 0) * matrix.get(2, 2))
            + (matrix.get(0, 1) * matrix.get(1, 2) * matrix.get(2, 0))
            + (matrix.get(0, 2) * matrix.get(1, 0) * matrix.get(2, 1))
            - (matrix.get(0, 2) * matrix.get(1, 1) * matrix.get(2, 0));

        assert_eq!(determinant, linear_algebra::determinant(&matrix).unwrap());
    }
}
