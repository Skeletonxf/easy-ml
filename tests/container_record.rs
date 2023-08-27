extern crate easy_ml;

#[cfg(test)]
mod container_record_tests {
    use easy_ml::tensors::Tensor;
    use easy_ml::tensors::views::TensorView;
    use easy_ml::tensors::indexing::ShapeIterator;
    use easy_ml::differentiation::{Record, RecordTensor, WengertList};

    #[test]
    fn test_subtraction_derivatives() {
        let list = WengertList::new();
        let x = RecordTensor::variables(
            &list,
            Tensor::from_fn([("r", 3), ("c", 3)], |[x, y]| x as f64 + y as f64)
        );
        let y = RecordTensor::variables(
            &list,
            Tensor::from_fn([("r", 3), ("c", 3)], |[x, y]| 3.0 + x as f64 - y as f64)
        );
        let subtraction = |x, y| x - y;
        // δ(lhs - rhs) / lhs = 1
        let subtraction_dx = |_x, _y| 1.0;
        // δ(lhs - rhs) / rhs = -1
        let subtraction_dy = |_x, _y| -1.0;
        let x_minus_y = x
            .binary(&y, subtraction, subtraction_dx, subtraction_dy);
        let also_x_minus_y = x
            .clone()
            .do_binary_left_assign(&y, subtraction, subtraction_dx, subtraction_dy);
        let also_also_x_minus_y = x
            .do_binary_right_assign(y.clone(), subtraction, subtraction_dx, subtraction_dy);

        let y_minus_x = y
            .binary(&x, subtraction, subtraction_dx, subtraction_dy);
        let also_y_minus_x = y
            .clone()
            .do_binary_left_assign(&x, subtraction, subtraction_dx, subtraction_dy);
        let also_also_y_minus_x = y
            .do_binary_right_assign(x.clone(), subtraction, subtraction_dx, subtraction_dy);

        let ops_x_minus_y = &x - &y;
        let ops_y_minus_x = &y - &x;

        let expected: Vec<_> = {
            let history = WengertList::new();
            vec![
                (0.0, 3.0), (1.0, 4.0), (2.0, 5.0),
                (1.0, 2.0), (2.0, 3.0), (3.0, 4.0),
                (2.0, 1.0), (3.0, 2.0), (4.0, 3.0)
            ].into_iter().map(|(x, y)| {
                let x = Record::variable(x, &history);
                let y = Record::variable(y, &history);
                let x_minus_y = x - y;
                let y_minus_x = y - x;
                let x_minus_y_derivatives = x_minus_y.derivatives();
                let x_minus_y_dx = x_minus_y_derivatives.at(&x);
                let x_minus_y_dy = x_minus_y_derivatives.at(&y);
                let y_minus_x_derivatives = y_minus_x.derivatives();
                let y_minus_x_dx = y_minus_x_derivatives.at(&x);
                let y_minus_x_dy = y_minus_x_derivatives.at(&y);
                (x_minus_y_dx, x_minus_y_dy, y_minus_x_dx, y_minus_x_dy)
            }).collect()
        };

        assert!(expected.iter().all(|(x_minus_y_dx, x_minus_y_dy, y_minus_x_dx, y_minus_x_dy)| {
            *x_minus_y_dx == 1.0 &&
            *x_minus_y_dy == -1.0 &&
            *y_minus_x_dx == -1.0 &&
            *y_minus_x_dy == 1.0
        }));

        for container in vec![x_minus_y, also_x_minus_y, also_also_x_minus_y, ops_x_minus_y] {
            for index in ShapeIterator::from(container.shape()) {
                let derivatives = container.derivatives_for(index).unwrap();
                let dx = derivatives.at_tensor_index(index, &x).unwrap();
                let dy = derivatives.at_tensor_index(index, &y).unwrap();
                assert_eq!(dx, 1.0);
                assert_eq!(dy, -1.0);
            }
        }

        for container in vec![y_minus_x, also_y_minus_x, also_also_y_minus_x, ops_y_minus_x] {
            for index in ShapeIterator::from(container.shape()) {
                let derivatives = container.derivatives_for(index).unwrap();
                let dx = derivatives.at_tensor_index(index, &x).unwrap();
                let dy = derivatives.at_tensor_index(index, &y).unwrap();
                assert_eq!(dx, -1.0);
                assert_eq!(dy, 1.0);
            }
        }
    }

    #[test]
    fn test_division_derivatives() {
        let list = WengertList::new();
        let x = RecordTensor::variables(
            &list,
            Tensor::from(
                [("r", 3), ("c", 3)],
                vec![
                    12.0, 15.0, 18.0,
                    360.0, 180.0, 90.0,
                    45.0, 240.0, 10.0
                ]
            )
        );
        let y = RecordTensor::variables(
            &list,
            Tensor::from(
                [("r", 3), ("c", 3)],
                vec![
                    4.0, 3.0, 6.0,
                    18.0, 10.0, 30.0,
                    5.0, 20.0, 5.0
                ]
            )
        );

        let x_div_y = x.elementwise_divide(&y);

        #[rustfmt::skip]
        // Normally we'd want a tolerance for float comparisons but the implementation should
        // always perform just a normal division so we should always get identical results
        // here comparing to the calculation with constants.
        assert_eq!(
            TensorView::from(&x_div_y).map(|(x, _)| x),
            Tensor::from(
                [("r", 3), ("c", 3)],
                vec![
                    12.0 / 4.0,   15.0 / 3.0,    18.0 / 6.0,
                    360.0 / 18.0, 180.0 / 10.0,  90.0 / 30.0,
                    45.0 / 5.0,   240.0 / 20.0,  10.0 / 5.0
                ]
            )
        );

        for container in vec![x_div_y] {
            let derivatives = container.derivatives().unwrap();
            let dx = derivatives.map(|d| d.at_tensor(&x));
            let dy = derivatives.map(|d| d.at_tensor(&y));

            // x / y, derivative with respect to x is 1 / x, rest are 0 because output and input
            // unrelated
            assert_eq!(
                dx,
                Tensor::from(
                    [("r", 3), ("c", 3)],
                    vec![
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, 1.0 / 18.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, 0.0, 1.0 / 10.0, 0.0, 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 30.0, 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 5.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 20.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 5.0 ]
                        )
                    ]
                )
            );

            // x / y, derivative with respect to y is -x / (y^2), rest are 0 because output and
            // input unrelated
            assert_eq!(
                dy,
                Tensor::from(
                    [("r", 3), ("c", 3)],
                    vec![
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ -12.0 / (4.0 * 4.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, -15.0 / (3.0 * 3.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, -18.0 / (6.0 * 6.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, -360.0 / (18.0 * 18.0), 0.0, 0.0, 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, 0.0, -180.0 / (10.0 * 10.0), 0.0, 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, 0.0, 0.0, -90.0 / (30.0 * 30.0), 0.0, 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -45.0 / (5.0 * 5.0), 0.0, 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -240.0 / (20.0 * 20.0), 0.0 ]
                        ),
                        Tensor::from(
                            [("r", 3), ("c", 3)],
                            vec![ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0 / (5.0 * 5.0) ]
                        )
                    ]
                )
            );
        }
    }
}
