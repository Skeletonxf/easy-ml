use crate::numeric::{Numeric, NumericRef};
use crate::numeric::extra::{Sqrt, Pi, Exp, Pow};

/**
 * A Gaussian probability density function of a normally distributed
 * random variable with expected value / mean μ, and variance σ^2.
 *
 */
struct Gaussian<T: Numeric> {
    mean: T,
    variance: T
}

impl <T: Numeric> Gaussian<T>
where for<'a> &'a T: NumericRef<T> {
    /**
     * Computes g(x) for some x
     * TODO
     * https://en.wikipedia.org/wiki/Gaussian_function
     */
    pub fn map(&self, x: &T) -> T
        where
            T: Pi + Exp<Output = T> + Pow<Output = T> + Sqrt<Output = T>,
            for<'a> &'a T: Sqrt<Output = T>,
            for<'a> T: Pow<&'a T, Output = T>, {
        let standard_deviation = (&self.variance).sqrt();
        let two = T::one() + T::one();
        let two_pi = T::pi() * &two;
        let fraction = T::one() / (standard_deviation * (two_pi.sqrt()));
        let exponent = (- T::one() / &two) * ((x - &self.mean) / &self.variance).pow(&two);
        fraction * exponent.exp()
    }
}

#[test]
fn test_equation() {
    let function: Gaussian<f64> = Gaussian {
        mean: 0.0,
        variance: 1.0,
    };
    // this is giving the likelihood of at each x for our zero mean gaussian
    println!("{},{},{},{},{}", function.map(&-1.0), function.map(&0.0), function.map(&1.0), function.map(&2.0), function.map(&3.0));
}
