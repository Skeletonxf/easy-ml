extern crate easy_ml;

#[cfg(test)]
mod tests {
    use easy_ml::differentiation::Trace;

    #[test]
    fn test_adding() {
        let a = Trace { number: 2.0, derivative: 1.0 };
        let b = Trace { number: -1.0, derivative: 1.0 };
        let _c = &a + &b;
        let _d = &a + b;
        let _e = a + &b;
        let _f = a + b;
        assert_eq!(_c, _d);
        assert_eq!(_e, _f);
        assert_eq!(_c, Trace { number: 1.0, derivative: 2.0 });
    }

    fn three_x_squared(x: Trace<f32>) -> Trace<f32> {
        x * x * Trace::constant(3.0)
    }

    fn three_x_squared_derivative(x: f32) -> f32 {
        // d 3(x^2) / dx == 6x
        6.0 * x
    }

    #[test]
    fn test_three_x_squared() {
        // Test the differentiation of the function 3(x^2) with respect to x
        let x = 1.5;
        let dx = three_x_squared(Trace { number: x, derivative: 1.0 });
        let also_dx = three_x_squared_derivative(1.5);
        assert_eq!(dx.derivative, also_dx);
    }

    #[test]
    fn test_four_x_cubed() {
        // Test the differentiation of the function 4(x^3) with respect to x
        let x = 0.75;
        let dx = Trace::derivative(|x| Trace::constant(4.0) * x * x * x, x);
        let also_dx = 12.0 * x * x;
        assert_eq!(dx, also_dx);
    }
}
