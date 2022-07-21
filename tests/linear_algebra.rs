extern crate easy_ml;

#[cfg(test)]
mod linear_algebra {
    use easy_ml::linear_algebra;
    use easy_ml::matrices::Matrix;
    use easy_ml::tensors::Tensor;

    use rand::{Rng, SeedableRng};

    #[test]
    fn check_determinant_2_by_2() {
        let matrix = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let a = matrix.get(0, 0);
        let b = matrix.get(0, 1);
        let c = matrix.get(1, 0);
        let d = matrix.get(1, 1);
        let determinant = (a * d) - (b * c);
        assert_eq!(
            determinant,
            linear_algebra::determinant::<f32>(&matrix).unwrap()
        );
    }

    #[test]
    fn check_determinant_3_by_3() {
        // use the example from wikipedia
        // https://en.wikipedia.org/wiki/Determinant#n_%C3%97_n_matrices
        #[rustfmt::skip]
        let matrix = Matrix::from(vec![
            vec![-1.0,  2.0,  3.0],
            vec![ 3.0,  4.0, -5.0],
            vec![ 5.0, -2.0,  3.0]]
        );

        let determinant = 0.0 + (matrix.get(0, 0) * matrix.get(1, 1) * matrix.get(2, 2))
            - (matrix.get(0, 0) * matrix.get(1, 2) * matrix.get(2, 1))
            - (matrix.get(0, 1) * matrix.get(1, 0) * matrix.get(2, 2))
            + (matrix.get(0, 1) * matrix.get(1, 2) * matrix.get(2, 0))
            + (matrix.get(0, 2) * matrix.get(1, 0) * matrix.get(2, 1))
            - (matrix.get(0, 2) * matrix.get(1, 1) * matrix.get(2, 0));

        assert_eq!(determinant, matrix.determinant().unwrap());
    }

    #[test]
    fn check_determinant_2_by_2_tensor() {
        let tensor = Tensor::from([("x", 2), ("y", 2)], vec![1.0, 2.0, 3.0, 4.0]);
        let x_y = tensor.index();
        let a = x_y.get([0, 0]);
        let b = x_y.get([0, 1]);
        let c = x_y.get([1, 0]);
        let d = x_y.get([1, 1]);
        let determinant = (a * d) - (b * c);
        assert_eq!(
            determinant,
            linear_algebra::determinant_tensor::<f32, _, _>(&tensor).unwrap()
        );
    }

    #[test]
    fn check_determinant_3_by_3_tensor() {
        // use the example from wikipedia
        // https://en.wikipedia.org/wiki/Determinant#n_%C3%97_n_matrices
        #[rustfmt::skip]
        let tensor = Tensor::from([("x", 3), ("y", 3)], vec![
            -1.0,  2.0,  3.0,
            3.0,  4.0, -5.0,
            5.0, -2.0,  3.0
        ]);
        let x_y = tensor.index();

        let determinant = 0.0 + (x_y.get([0, 0]) * x_y.get([1, 1]) * x_y.get([2, 2]))
            - (x_y.get([0, 0]) * x_y.get([1, 2]) * x_y.get([2, 1]))
            - (x_y.get([0, 1]) * x_y.get([1, 0]) * x_y.get([2, 2]))
            + (x_y.get([0, 1]) * x_y.get([1, 2]) * x_y.get([2, 0]))
            + (x_y.get([0, 2]) * x_y.get([1, 0]) * x_y.get([2, 1]))
            - (x_y.get([0, 2]) * x_y.get([1, 1]) * x_y.get([2, 0]));

        assert_eq!(determinant, tensor.determinant().unwrap());
    }

    #[test]
    fn inverse_1_by_1() {
        let matrix = Matrix::from_scalar(3.0);
        let inverse = linear_algebra::inverse::<f32>(&matrix).unwrap();
        let absolute_difference = (inverse.scalar() - (1.0 / 3.0)).abs();
        assert!(absolute_difference <= std::f32::EPSILON);
    }

    #[test]
    fn inverse_1_by_1_tensor() {
        let matrix = Tensor::from([("r", 1), ("c", 1)], vec![3.0]);
        let inverse = linear_algebra::inverse_tensor::<f32, _, _>(&matrix).unwrap();
        let absolute_difference = (inverse.iter().next().unwrap() - (1.0 / 3.0)).abs();
        assert!(absolute_difference <= std::f32::EPSILON);
    }

    #[test]
    fn inverse_2_by_2() {
        let matrix = Matrix::from(vec![vec![4.0, 7.0], vec![2.0, 6.0]]);
        let inverse = matrix.inverse().unwrap();
        // we use the example from https://www.mathsisfun.com/algebra/matrix-inverse.html
        let answer = Matrix::from(vec![vec![0.6, -0.7], vec![-0.2, 0.4]]);
        for row in 0..answer.rows() {
            for column in 0..answer.columns() {
                let absolute_difference: f32 = inverse.get(row, column) - answer.get(row, column);
                assert!(absolute_difference.abs() <= std::f32::EPSILON);
            }
        }
        // multiplying the inverse and original should yeild the identity matrix
        let identity = &matrix * &inverse;
        let answer = Matrix::diagonal(1.0, matrix.size());
        for row in 0..answer.rows() {
            for column in 0..answer.columns() {
                let absolute_difference: f32 = identity.get(row, column) - answer.get(row, column);
                assert!(absolute_difference.abs() <= std::f32::EPSILON);
            }
        }
    }

    #[test]
    fn inverse_2_by_2_tensor() {
        let matrix = Tensor::from([("row", 2), ("column", 2)], vec![4.0, 7.0, 2.0, 6.0]);
        let inverse = matrix.inverse().unwrap();
        // we use the example from https://www.mathsisfun.com/algebra/matrix-inverse.html
        let answer = Tensor::from([("row", 2), ("column", 2)], vec![0.6, -0.7, -0.2, 0.4]);
        for (expected, actual) in answer.iter().zip(inverse.iter()) {
            let absolute_difference: f32 = actual - expected;
            assert!(absolute_difference.abs() <= std::f32::EPSILON);
        }
        // multiplying the inverse and original should yeild the identity matrix
        let identity = &matrix * &inverse;
        let also_identity = Tensor::from([("row", 2), ("column", 2)], vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(identity.shape(), also_identity.shape());
        for (x, y) in identity.iter().zip(also_identity.iter()) {
            assert!((x - y).abs() <= std::f32::EPSILON);
        }
    }

    #[test]
    fn inverse_2_by_2_not_inversible() {
        let matrix = Matrix::from(vec![vec![3.0, 4.0], vec![6.0, 8.0]]);
        let inverse = matrix.inverse();
        assert!(inverse.is_none());
    }

    #[test]
    fn inverse_2_by_2_not_inversible_tensor() {
        let matrix = Tensor::from([("rows", 2), ("columns", 2)], vec![3.0, 4.0, 6.0, 8.0]);
        let inverse = matrix.inverse();
        assert!(inverse.is_none());
    }

    #[test]
    #[rustfmt::skip]
    fn test_covariance() {
        let matrix: Matrix<f32> = Matrix::from(vec![
                vec![  1.0,  1.0, -1.0],
                vec![ -1.0, -1.0,  1.0],
                vec![ -1.0, -1.0,  1.0],
                vec![  1.0,  1.0, -1.0]]);
        assert_eq!(
            linear_algebra::covariance_column_features::<f32>(&matrix),
            Matrix::from(vec![
                vec![ 1.0,   1.0, -1.0 ],
                vec![ 1.0,   1.0, -1.0 ],
                vec![ -1.0, -1.0,  1.0 ]]
            )
        );
    }

    #[test]
    #[rustfmt::skip]
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
        #[rustfmt::skip]
        let matrix = Matrix::from(vec![
            vec![ 18.0, 22.0,  54.0,  42.0 ],
            vec![ 22.0, 70.0,  86.0,  62.0 ],
            vec![ 54.0, 86.0, 174.0, 134.0 ],
            vec![ 42.0, 62.0, 134.0, 106.0 ]]);
        let lower_triangular = linear_algebra::cholesky_decomposition::<f64>(&matrix).unwrap();
        let recovered = &lower_triangular * lower_triangular.transpose();
        #[rustfmt::skip]
        let expected = Matrix::from(vec![
            vec![  4.24264, 0.0,     0.0,     0.0     ],
            vec![  5.18545, 6.56591, 0.0,     0.0     ],
            vec![ 12.72792, 3.04604, 1.64974, 0.0     ],
            vec![  9.89949, 1.62455, 1.84971, 1.39262 ]]);
        let absolute_difference: f64 = lower_triangular
            .column_major_iter()
            .zip(expected.column_major_iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        println!("absolute_difference: {}", absolute_difference);
        assert!(absolute_difference < 0.0001);
        let absolute_difference: f64 = matrix
            .column_major_iter()
            .zip(recovered.column_major_iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(absolute_difference < 0.0001);
    }

    #[test]
    fn test_mean() {
        assert_eq!(
            linear_algebra::mean(vec![2.0, -2.0, 0.0, 1.0].drain(..)),
            0.25
        );
    }

    #[test]
    fn test_variance() {
        // test is performed by using the alternate formula and avoiding linear_algebra::mean
        // because linear_algebra::variance uses linear_algebra::mean in its implementation
        let data = vec![2.0, -2.0, 0.0, 1.0];
        let squared_mean = data.iter().map(|x| x * x).sum::<f64>() / 4.0;
        let mean_squared = (data.iter().cloned().sum::<f64>() / 4.0).powi(2);
        assert_eq!(
            linear_algebra::variance(data.iter().cloned()),
            squared_mean - mean_squared
        );
    }

    #[test]
    fn test_softmax_positive() {
        let input: Vec<f64> = vec![2.0, 3.0, 4.0, 2.0];
        let softmax = linear_algebra::softmax(input.iter().cloned());
        let expected: Vec<f64> = vec![
            2.0_f64.exp() / (2.0_f64.exp() + 3.0_f64.exp() + 4.0_f64.exp() + 2.0_f64.exp()),
            3.0_f64.exp() / (2.0_f64.exp() + 3.0_f64.exp() + 4.0_f64.exp() + 2.0_f64.exp()),
            4.0_f64.exp() / (2.0_f64.exp() + 3.0_f64.exp() + 4.0_f64.exp() + 2.0_f64.exp()),
            2.0_f64.exp() / (2.0_f64.exp() + 3.0_f64.exp() + 4.0_f64.exp() + 2.0_f64.exp()),
        ];
        assert_eq!(softmax.len(), expected.len());
        let absolute_difference: f64 = softmax
            .iter()
            .zip(expected.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(absolute_difference < 0.0001);
    }

    #[test]
    fn test_softmax_negative() {
        let input: Vec<f64> = vec![-2.0, -7.0, -9.0];
        let softmax = linear_algebra::softmax(input.iter().cloned());
        let expected: Vec<f64> = vec![
            (-2.0_f64).exp() / ((-2.0_f64).exp() + (-7.0_f64).exp() + (-9.0_f64).exp()),
            (-7.0_f64).exp() / ((-2.0_f64).exp() + (-7.0_f64).exp() + (-9.0_f64).exp()),
            (-9.0_f64).exp() / ((-2.0_f64).exp() + (-7.0_f64).exp() + (-9.0_f64).exp()),
        ];
        assert_eq!(softmax.len(), expected.len());
        let absolute_difference: f64 = softmax
            .iter()
            .zip(expected.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(absolute_difference < 0.0001);
    }

    #[test]
    fn test_softmax_mixed() {
        let input: Vec<f64> = vec![-2.0, 2.0];
        let softmax = linear_algebra::softmax(input.iter().cloned());
        let expected: Vec<f64> = vec![
            (-2.0_f64).exp() / ((-2.0_f64).exp() + (2.0_f64).exp()),
            (2.0_f64).exp() / ((-2.0_f64).exp() + (2.0_f64).exp()),
        ];
        assert_eq!(softmax.len(), expected.len());
        let absolute_difference: f64 = softmax
            .iter()
            .zip(expected.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(absolute_difference < 0.0001);
    }

    #[test]
    fn test_softmax_empty_list() {
        let input: Vec<f32> = Vec::with_capacity(0);
        assert_eq!(linear_algebra::softmax(input.iter().cloned()), input);
    }

    #[test]
    fn test_qr_decomposition_3_by_3() {
        // Test input from https://rosettacode.org/wiki/QR_decomposition
        #[rustfmt::skip]
        let input = Matrix::from(vec![
            vec![ 12.0, -51.0,   4.0 ],
            vec![  6.0, 167.0, -68.0 ],
            vec![ -4.0,  24.0, -41.0 ],
        ]);
        let output = linear_algebra::qr_decomposition::<f64>(&input).unwrap();
        println!("Q:\n{}", output.q);
        println!("R:\n{}", output.r);
        #[rustfmt::skip]
        let q = Matrix::from(vec![
            vec![ -0.857,  0.394,  0.331 ],
            vec![ -0.429, -0.903, -0.034 ],
            vec![  0.286, -0.171,  0.943 ],
        ]);
        #[rustfmt::skip]
        let r = Matrix::from(vec![
            vec![ -14.000,  -21.000,   14.000 ],
            vec![  -0.000, -175.000,   70.000 ],
            vec![  -0.000,    0.0000, -35.000 ],
        ]);
        assert_eq!(q.size(), output.q.size());
        assert_eq!(r.size(), output.r.size());
        assert!(q
            .row_major_iter()
            .zip(output.q.row_major_iter())
            .all(|(expected, actual)| expected - actual < 0.001));
        assert!(r
            .row_major_iter()
            .zip(output.r.row_major_iter())
            .all(|(expected, actual)| expected - actual < 0.001));
    }

    #[test]
    fn test_qr_decomposition_fuzzing() {
        let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(55);
        for (r, c) in &[(2, 2), (3, 2), (3, 3), (4, 4), (5, 3), (5, 5), (6, 6)] {
            let mut data = vec![0.0; r * c];
            random_generator.fill(data.as_mut_slice());
            let matrix = Matrix::from_flat_row_major((*r, *c), data);
            let output = linear_algebra::qr_decomposition::<f64>(&matrix).unwrap();
            let q = output.q;
            let r = output.r;
            // Q should be orthogonal
            let identity = q.transpose() * &q;
            println!("Q:\n{}\nR:\n{}\nA:\n{}", q, r, matrix);
            println!("Q^TQ:\n{}", identity);
            for row in 0..identity.rows() {
                for column in 0..identity.columns() {
                    if row == column {
                        // diagonal should be 1
                        assert!((1.0 - identity.get(row, column)).abs() < 0.00001);
                    } else {
                        // non diagonal should be 0
                        assert!(identity.get(row, column).abs() < 0.00001);
                    }
                }
            }
            // All entries below diagonal on R should be zero
            for row in 0..r.rows() {
                for column in 0..r.columns() {
                    if row > column {
                        assert!(r.get(row, column).abs() < 0.00001);
                    }
                }
            }
            // QR should return the input
            let a = q * r;
            println!("A:\n{}", matrix);
            println!("Reconstructed A:\n{}", a);
            assert!(a
                .row_major_iter()
                .zip(matrix.row_major_iter())
                .map(|(x, y)| x - y)
                .all(|x| x.abs() < 0.00001));
        }
    }
}
