extern crate easy_ml;

#[cfg(test)]
mod forward_tests {
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

    use easy_ml::numeric::Numeric;
    // f(x) = (x^5 + x^3 - 1/x) - x
    // df(x)/dx = 5x^4 + 3x^2 + (1/x^2) - 1
    fn f<T: Numeric + Copy>(x: T) -> T {
        ((x * x * x * x * x) + (x * x * x) - (T::one() / x)) - x
    }

    #[test]
    fn test_numeric_substitution() {
        // Tests that the same function written for a float generically with Numeric
        // can also be used with Trace<float> and both uses compute the same result.
        let x = -0.75;
        let result = f(Trace::variable(x));
        let dx = result.derivative;
        let y = result.number;
        let also_y = f(-0.75);
        assert_eq!(y, also_y);
        let also_dx = (5.0 * x * x * x * x) + (3.0 * x * x) + (1.0 / (x * x)) - 1.0;
        assert_eq!(dx, also_dx);
    }
}


#[cfg(test)]
mod reverse_tests {
    use easy_ml::differentiation::Record;
    use easy_ml::differentiation::WengertList;

    fn x_cubed(x: Record<f32>) -> Record<f32> {
        &(&x * &x) * &x
    }

    fn x_cubed_derivative(x: f32) -> f32 {
        // d (x^3) / dx == 3(x^2)
        3.0 * x * x
    }

    #[test]
    fn test_x_cubed() {
        // Test the differentiation of the function (x^3) with respect to x
        let list = WengertList::new();
        let x = list.record(1.5);
        let index = x.index;
        let y = x_cubed(x);
        let gradients = y.gradients();
        let dx = gradients[index];
        let also_dx = x_cubed_derivative(1.5);
        assert_eq!(dx, also_dx);
    }
}
