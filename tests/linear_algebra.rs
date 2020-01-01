extern crate easy_ml;

#[cfg(test)]
mod tests {
    use easy_ml::matrices::Matrix;
    use easy_ml::linear_algebra;

    #[test]
    fn check_determinant_2_by_2() {
        let matrix: Matrix<f32> = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
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

    #[test]
    fn inverse_1_by_1() {
        let matrix = Matrix::unit(3.0);
        let inverse = linear_algebra::inverse(&matrix).unwrap();
        let absolute_difference = inverse.get(0, 0) - (1.0 / 3.0);
        assert!(absolute_difference <= std::f32::EPSILON);
    }

    #[test]
    fn inverse_2_by_2() {
        let matrix = Matrix::from(vec![
            vec![ 4.0, 7.0 ],
            vec![ 2.0, 6.0 ]]);
        let inverse = linear_algebra::inverse(&matrix).unwrap();
        // we use the example from https://www.mathsisfun.com/algebra/matrix-inverse.html
        let answer = Matrix::from(vec![
            vec![ 0.6, -0.7 ],
            vec![ -0.2, 0.4 ]]);
        for row in 0..answer.rows() {
            for column in 0..answer.columns() {
                let absolute_difference = inverse.get(row, column) - answer.get(row, column);
                assert!(absolute_difference <= std::f32::EPSILON);
            }
        }
        // multiplying the inverse and original should yeild the identity matrix
        let identity = &matrix * &inverse;
        let answer: Matrix<f32> = Matrix::identity(matrix.size());
        for row in 0..answer.rows() {
            for column in 0..answer.columns() {
                let absolute_difference = identity.get(row, column) - answer.get(row, column);
                assert!(absolute_difference <= std::f32::EPSILON);
            }
        }
    }

    #[test]
    fn inverse_2_by_2_not_inversible() {
        let matrix = Matrix::from(vec![
            vec![ 3.0, 4.0 ],
            vec![ 6.0, 8.0 ]]);
        let inverse = linear_algebra::inverse(&matrix);
        assert!(inverse.is_none());
    }
}
