extern crate easy_ml;

#[cfg(test)]
mod container_record_tests {
    use easy_ml::tensors::Tensor;
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

    // TODO: Test division works as expected too
}
