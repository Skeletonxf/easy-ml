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

    #[test]
    fn test_constant_lifting() {
        let x = 0.23;
        let y = (Trace::variable(x) + 0.3) * 1.2;
        let also_y = (Trace::variable(x) + Trace::constant(0.3)) * Trace::constant(1.2);
        assert_eq!(y.number, also_y.number);
        assert_eq!(y.derivative, also_y.derivative);
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
        let x = list.variable(1.5);
        let index = x.index;
        let y = x_cubed(x);
        let derivatives = y.derivatives();
        let dx = derivatives[index];
        let also_dx = x_cubed_derivative(1.5);
        assert_eq!(dx, also_dx);
    }

    #[test]
    fn test_four_x_cubed() {
        // Test the differentiation of the function 4(x^3) with respect to x
        let list = WengertList::new();
        let x = Record::variable(0.75, &list);
        let y = Record::constant(4.0) * x * x * x;
        let derivatives = y.derivatives();
        let dx = derivatives[x.index];
        let also_dx = 12.0 * 0.75 * 0.75;
        assert_eq!(dx, also_dx);
    }

    #[test]
    fn test_adding_and_multiplying_constants() {
        // Test the differentiation of the function 2.3(x+0.66)x with respect to x
        let list = WengertList::new();
        let x = Record::variable(0.34, &list);
        let y = (Record::constant(2.3) * (x + 0.66)) * x;
        let derivatives = y.derivatives();
        let dx = derivatives[x.index];
        // https://www.wolframalpha.com/input/?i=d%282.3*%28x%2B0.66%29*x%29%2Fdx
        // dx = 4.6x + 1.518
        let also_dx = (4.6 * 0.34) + 1.518;
        assert_eq!(dx, also_dx);
    }

    #[test]
    fn test_division_numerator() {
        // Test the differentiation of the function 1 - x/2.5 with respect to x
        let list = WengertList::new();
        let x = Record::variable(1.65, &list);
        let y = Record::constant(1.0) - (x / 2.5);
        let derivatives = y.derivatives();
        let dx = derivatives[x.index];
        // https://www.wolframalpha.com/input/?i=d%281-x%2F2.5%29%2Fdx
        // dx = -0.4
        assert_eq!(dx, -0.4);
    }

    #[test]
    fn test_division_denominator() {
        // Test the differentiation of the function 1 - 2.5/x with respect to x
        let list = WengertList::new();
        let x = Record::variable(0.25, &list);
        let y = Record::constant(2.5) / x;
        let derivatives = y.derivatives();
        let dx = derivatives[x.index];
        // https://www.wolframalpha.com/input/?i=d%281-%282.5%2Fx%29%29%2Fdx
        // dx = -(2.5 / x^2)
        let also_dx = -2.5 / (0.25 * 0.25);
        assert_eq!(dx, also_dx);
        let z = Record::constant(1.0) - y;
        let derivatives = z.derivatives();
        let dx = derivatives[x.index];
        // dx = 2.5 / x^2
        let also_dx = 2.5 / (0.25 * 0.25);
        assert_eq!(dx, also_dx);
    }
}
