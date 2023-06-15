extern crate easy_ml;

#[cfg(test)]
mod container_record_tests {
    use easy_ml::tensors::Tensor;
    use easy_ml::differentiation::{RecordTensor, WengertList};

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

        // TODO: Inspecting derivatives of RecordTensor
        // Accessing values of RecordTensor
        // Validate that the derivatives were calculated the same for all 3 API methods
    }

    // TODO: Test division works as expected too
}
