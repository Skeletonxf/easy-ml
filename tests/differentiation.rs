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

    use crate::easy_ml::numeric::extra::Pow;

    #[test]
    fn test_pow_equivalents_dx() {
        let mut x_derivatives = Vec::with_capacity(2);
        let x = 1.35;
        let y = 2.5;
        {
            let x = Trace::variable(x);
            let y = Trace::constant(y);
            let z = x.pow(y);
            x_derivatives.push(z.derivative);
        }
        {
            let x = Trace::variable(x);
            let z = x.pow(y);
            x_derivatives.push(z.derivative);
        }
        // d(x^y)/dx = y*x^(y-1)
        let also_dx = y * x.pow(y - 1.0);
        assert!(x_derivatives.iter().all(|&dx| dx == also_dx));
    }

    use crate::easy_ml::numeric::extra::Ln;

    #[test]
    fn test_pow_equivalents_dy() {
        let mut y_derivatives = Vec::with_capacity(2);
        let x = 1.35;
        let y = 2.5;
        {
            let x = Trace::constant(x);
            let y = Trace::variable(y);
            let z = x.pow(y);
            y_derivatives.push(z.derivative);
        }
        {
            let y = Trace::variable(y);
            let z = x.pow(y);
            y_derivatives.push(z.derivative);
        }
        // d(x^y)/dy = x^y * ln(x)
        let also_dy = x.pow(y) * x.ln();
        assert!(y_derivatives.iter().all(|&dy| dy == also_dy));
    }

    #[test]
    fn test_ln_gradient_descent() {
        // ln(x) approaches -inf as x goes to 0
        let mut x = Trace::variable(3.0);
        let steps = 10;
        let mut epochs = Vec::with_capacity(steps + 1);
        epochs.push(x.number);
        for _ in 0..steps {
            let y = x.ln();
            x = Trace::variable(x.number - (0.1 * y.derivative));
            epochs.push(x.number);
        }
        // check every epoch gave a smaller x value
        assert!(epochs.iter().enumerate().fold(true, |_, (i, &x)| {
            if i > 0 {
                x < epochs[i - 1]
            } else {
                true
            }
        }));
        // check x never went negative
        assert!(x.number > 0.0);
    }

    use crate::easy_ml::numeric::extra::Exp;

    #[test]
    fn test_exp_gradient_descent() {
        // e^x approaches 0 as x goes to -inf
        let mut x = Trace::variable(3.0);
        let steps = 10;
        let mut epochs = Vec::with_capacity(steps + 1);
        epochs.push(x.number);
        for _ in 0..steps {
            let y = x.exp();
            x = Trace::variable(x.number - (0.1 * y.derivative));
            epochs.push(x.number);
        }
        // check every epoch gave a smaller x value
        assert!(epochs.iter().enumerate().fold(true, |_, (i, &x)| {
            if i > 0 {
                x < epochs[i - 1]
            } else {
                true
            }
        }));
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
        let y = x_cubed(x);
        let derivatives = y.derivatives();
        let dx = derivatives[&x];
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
        let dx = derivatives[&x];
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
        let dx = derivatives[&x];
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
        let dx = derivatives[&x];
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
        let dx = derivatives[&x];
        // https://www.wolframalpha.com/input/?i=d%281-%282.5%2Fx%29%29%2Fdx
        // dx = -(2.5 / x^2)
        let also_dx = -2.5 / (0.25 * 0.25);
        assert_eq!(dx, also_dx);
        let z = Record::constant(1.0) - y;
        let derivatives = z.derivatives();
        let dx = derivatives[&x];
        // dx = 2.5 / x^2
        let also_dx = 2.5 / (0.25 * 0.25);
        assert_eq!(dx, also_dx);
    }

    #[test]
    fn test_sum() {
        // Test summation gives the same derivatives as explicit adding
        let list = WengertList::new();
        let x = Record::variable(0.5, &list);
        let y = Record::variable(2.0, &list);
        let z = Record::variable(2.0, &list);
        let result = [ x * 3.0, (y * y) / z, (z * y) - 5.0 ].iter().cloned().sum::<Record<f32>>();
        let derivatives = result.derivatives();
        let dx = derivatives[&x];
        let dy = derivatives[&y];
        let dz = derivatives[&z];
        let also_list = WengertList::<f32>::new();
        let also_x = Record::variable(0.5, &also_list);
        let also_y = Record::variable(2.0, &also_list);
        let also_z = Record::variable(2.0, &also_list);
        let also_result = (also_x * 3.0) + ((also_y * also_y) / also_z) + ((also_z * also_y) - 5.0);
        let also_derivatives = also_result.derivatives();
        let also_dx = also_derivatives[&also_x];
        let also_dy = also_derivatives[&also_y];
        let also_dz = also_derivatives[&also_z];
        assert_eq!(dx, also_dx);
        assert_eq!(dy, also_dy);
        assert_eq!(dz, also_dz);
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
        // can also be used with Record<float> and both uses compute the same result.
        let list = WengertList::new();
        let x = -0.75;
        let result = f(Record::variable(x, &list));
        let derivatives: Vec<_> = result.derivatives().into();
        let dx = derivatives[0]; // index of the Record wrapped x is 0
        let y = result.number;
        let also_y = f(-0.75);
        assert_eq!(y, also_y);
        let also_dx = (5.0 * x * x * x * x) + (3.0 * x * x) + (1.0 / (x * x)) - 1.0;
        assert_eq!(dx, also_dx);
    }

    #[test]
    fn test_constant_lifting_and_reusing_list() {
        let list = WengertList::new();
        let _x = 0.23;
        let x = Record::variable(_x, &list);
        let y = (x + 0.3) * 1.2;
        let also_x = Record::variable(_x, &list);
        let also_y = (also_x + Record::constant(0.3)) * Record::constant(1.2);
        let derivatives = y.derivatives();
        let also_derivatives = also_y.derivatives();
        assert_eq!(y.number, also_y.number);
        assert_eq!(derivatives[&x], also_derivatives[&also_x]);
    }

    use crate::easy_ml::numeric::extra::Pow;

    #[test]
    fn test_pow_equivalents_dx() {
        let mut x_derivatives = Vec::with_capacity(3);
        let x = 1.35;
        let y = 2.5;
        {
            let list = WengertList::new();
            let x = Record::variable(x, &list);
            let y = Record::variable(y, &list);
            let z = x.pow(y);
            x_derivatives.push(z.derivatives()[&x]);
        }
        {
            let list = WengertList::new();
            let x = Record::variable(x, &list);
            let z = x.pow(y);
            x_derivatives.push(z.derivatives()[&x]);
        }
        {
            let list = WengertList::new();
            let x = Record::variable(x, &list);
            let y = Record::constant(y);
            let z = x.pow(y);
            x_derivatives.push(z.derivatives()[&x]);
        }
        // d(x^y)/dx = y*x^(y-1)
        let also_dx = y * x.pow(y - 1.0);
        assert!(x_derivatives.iter().all(|&dx| dx == also_dx));
    }

    use crate::easy_ml::numeric::extra::Ln;

    #[test]
    fn test_pow_equivalents_dy() {
        let mut y_derivatives = Vec::with_capacity(3);
        let x = 1.35;
        let y = 2.5;
        {
            let list = WengertList::new();
            let x = Record::variable(x, &list);
            let y = Record::variable(y, &list);
            let z = x.pow(y);
            y_derivatives.push(z.derivatives()[&y]);
        }
        {
            let list = WengertList::new();
            let y = Record::variable(y, &list);
            let z = x.pow(y);
            y_derivatives.push(z.derivatives()[&y]);
        }
        {
            let list = WengertList::new();
            let x = Record::constant(x);
            let y = Record::variable(y, &list);
            println!("made it to computing z");
            let z = x.pow(y);
            println!("computed z");
            y_derivatives.push(z.derivatives()[&y]);
            println!("computed dy");
        }
        // d(x^y)/dy = x^y * ln(x)
        let also_dy = x.pow(y) * x.ln();
        assert!(y_derivatives.iter().all(|&dy| dy == also_dy));
    }

    #[test]
    fn test_ln_gradient_descent() {
        let list = WengertList::new();
        // ln(x) approaches -inf as x goes to 0
        let mut x = Record::variable(3.0, &list);
        let steps = 10;
        let mut epochs = Vec::with_capacity(steps + 1);
        epochs.push(x.number);
        for _ in 0..steps {
            let y = x.ln();
            x = Record::variable(x.number - (0.1 * y.derivatives()[&x]), &list);
            epochs.push(x.number);
            list.clear();
            x.reset();
        }
        // check every epoch gave a smaller x value
        assert!(epochs.iter().enumerate().fold(true, |_, (i, &x)| {
            if i > 0 {
                x < epochs[i - 1]
            } else {
                true
            }
        }));
        // check x never went negative
        assert!(x.number > 0.0);
    }

    use crate::easy_ml::numeric::extra::Exp;

    #[test]
    fn test_exp_gradient_descent() {
        let list = WengertList::new();
        // e^x approaches 0 as x goes to -inf
        let mut x = Record::variable(3.0, &list);
        let steps = 10;
        let mut epochs = Vec::with_capacity(steps + 1);
        epochs.push(x.number);
        for _ in 0..steps {
            let y = x.exp();
            x = Record::variable(x.number - (0.1 * y.derivatives()[&x]), &list);
            epochs.push(x.number);
            list.clear();
            x.reset();
        }
        // check every epoch gave a smaller x value
        assert!(epochs.iter().enumerate().fold(true, |_, (i, &x)| {
            if i > 0 {
                x < epochs[i - 1]
            } else {
                true
            }
        }));
    }
}
