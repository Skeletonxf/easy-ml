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
        assert_eq!(determinant, linear_algebra::determinant::<f32>(&matrix).unwrap());
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

        assert_eq!(determinant, matrix.determinant().unwrap());
    }

    #[test]
    fn inverse_1_by_1() {
        let matrix = Matrix::unit(3.0);
        let inverse = linear_algebra::inverse::<f32>(&matrix).unwrap();
        let absolute_difference = inverse.get(0, 0) - (1.0 / 3.0);
        assert!(absolute_difference <= std::f32::EPSILON);
    }

    #[test]
    fn inverse_2_by_2() {
        let matrix = Matrix::from(vec![
            vec![ 4.0, 7.0 ],
            vec![ 2.0, 6.0 ]]);
        let inverse = matrix.inverse().unwrap();
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
        let inverse = matrix.inverse();
        assert!(inverse.is_none());
    }

    #[test]
    fn test_covariance() {
        let matrix: Matrix<f32> = Matrix::from(vec![
                vec![  1.0,  1.0, -1.0],
                vec![ -1.0, -1.0,  1.0],
                vec![ -1.0, -1.0,  1.0],
                vec![  1.0,  1.0, -1.0]]);
        assert_eq!(linear_algebra::covariance::<f32>(&matrix).unwrap(),
            Matrix::from(vec![
                vec![ 1.0,  1.0, -1.0 ],
                vec![ 1.0,  1.0, -1.0 ],
                vec![ -1.0, -1.0, 1.0 ]]));
    }

    #[test]
    fn test_cholesky_decomposition_3_by_3() {
        // some examples are outlined at https://rosettacode.org/wiki/Cholesky_decomposition
        // they form the test cases here
        let matrix = Matrix::from(vec![
            vec![ 25.0, 15.0, -5.0 ],
            vec![ 15.0, 18.0,  0.0 ],
            vec![ -5.0,  0.0, 11.0 ]]);
        let lower_triangular = linear_algebra::cholesky_decomposition::<f32>(&matrix).unwrap();
        let recovered = &lower_triangular * lower_triangular.transpose();
        assert_eq!(lower_triangular, Matrix::from(vec![
            vec![ 5.0, 0.0, 0.0 ],
            vec![ 3.0, 3.0, 0.0 ],
            vec![-1.0, 1.0, 3.0 ]]));
        assert_eq!(matrix, recovered);
    }

    #[test]
    fn test_cholesky_decomposition_4_by_4() {
        // some examples are outlined at https://rosettacode.org/wiki/Cholesky_decomposition
        // they form the test cases here
        // this test case requires a lot of decimal representation so
        // is checked for approximation rather than exact value
        let matrix = Matrix::from(vec![
            vec![ 18.0, 22.0,  54.0,  42.0 ],
            vec![ 22.0, 70.0,  86.0,  62.0 ],
            vec![ 54.0, 86.0, 174.0, 134.0 ],
            vec![ 42.0, 62.0, 134.0, 106.0 ]]);
        let lower_triangular = linear_algebra::cholesky_decomposition::<f64>(&matrix).unwrap();
        let recovered = &lower_triangular * lower_triangular.transpose();
        let expected = Matrix::from(vec![
            vec![  4.24264, 0.0,     0.0,     0.0     ],
            vec![  5.18545, 6.56591, 0.0,     0.0     ],
            vec![ 12.72792, 3.04604, 1.64974, 0.0     ],
            vec![  9.89949, 1.62455, 1.84971, 1.39262 ]]);
        let absolute_difference: f64 = lower_triangular.column_major_iter()
            .zip(expected.column_major_iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        println!("absolute_difference: {}", absolute_difference);
        assert!(absolute_difference < 0.0001);
        let absolute_difference: f64 = matrix.column_major_iter()
            .zip(recovered.column_major_iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(absolute_difference < 0.0001);
    }
}
