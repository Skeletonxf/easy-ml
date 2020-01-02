/*!
 * Models of distributions that samples can be drawn from.
 */

use crate::numeric::{Numeric, NumericRef};
use crate::numeric::extra::{Sqrt, Pi, Exp, Pow, Ln, Sin, Cos};

// TODO: make non exhaustive and provide constructor function

/**
 * A Gaussian probability density function of a normally distributed
 * random variable with expected value / mean μ, and variance σ^2.
 *
 * See: [https://en.wikipedia.org/wiki/Gaussian_function](https://en.wikipedia.org/wiki/Gaussian_function)
 */
pub struct Gaussian<T: Numeric> {
    pub mean: T,
    pub variance: T
}

impl <T: Numeric> Gaussian<T>
where for<'a> &'a T: NumericRef<T> {
    /**
     * Computes g(x) for some x, the probability density of a normally
     * distributed random variable x, or in other words how likely x is
     * to be drawn from this normal distribution.
     *
     * g(x) is largest for x equal to this distribution's mean and
     * g(x) will tend towards zero as x is further from this distribution's
     * mean, at a rate corresponding to this distribution's variance.
     */
    pub fn map(&self, x: &T) -> T
        where
            T: Pi + Exp<Output = T> + Pow<Output = T> + Sqrt<Output = T>,
            for<'a> &'a T: Sqrt<Output = T>,
            for<'a> T: Pow<&'a T, Output = T>, {
        let standard_deviation = (&self.variance).sqrt();
        let two = T::one() + T::one();
        let two_pi = &two * T::pi();
        let fraction = T::one() / (standard_deviation * (&two_pi.sqrt()));
        let exponent = (- T::one() / &two) * ((x - &self.mean) / &self.variance).pow(&two);
        fraction * exponent.exp()
    }

    /**
     * Given a source of random variables in the uniformly distributed
     * range [0, 1] inclusive, draws up to `max_samples` of independent
     * random numbers according to this Gaussian distribution's mean and
     * variance using the Box-Muller transform:
     *
     * [https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
     *
     * The source of random variables must provide as many values
     * as `max_samples` if `max_samples` is even, and one more than `max_samples`
     * if `max_samples` is odd. If fewer are provided the returned
     * list of samples will not contain as many elements as max_samples.
     *
     * As all randomness is provided to this method, this code is deterministic
     * and will always compute the same samples given the same random source
     * of numbers.
     */
    pub fn draw<I>(&self, source: &mut I, max_samples: usize) -> Vec<T>
        where
            T: Pi + Sqrt<Output = T> + Cos<Output = T> + Sin<Output = T>,
            I: Iterator<Item = T>,
            for<'a> &'a T: Ln<Output = T>,
            for<'a> &'a T: Sqrt<Output = T>, {
        let two = T::one() + T::one();
        let minus_two = - &two;
        let two_pi = &two * T::pi();
        let mut samples = Vec::with_capacity(max_samples);
        let standard_deviation = (&self.variance).sqrt();
        // keep drawing samples from this normal Gaussian distribution
        // until either the iterator runs out or we reach the max_samples
        // limit
        while samples.len() < max_samples {
            match self.generate_pair(source) {
                Some((u, v)) => {
                    // these computations convert two samples from the inclusive 0 - 1
                    // range to two samples of a normal distribution with with
                    // μ = 0 and σ = 1.
                    let z1 = (&minus_two * &u.ln()).sqrt() * ((&two_pi * &v).cos());
                    let z2 = (&minus_two * &u.ln()).sqrt() * ((&two_pi * &v).sin());
                    // now we scale to the mean and variance for this Gaussian
                    let sample1 = (z1 * &standard_deviation) + &self.mean;
                    let sample2 = (z2 * &standard_deviation) + &self.mean;
                    samples.push(sample1);
                    samples.push(sample2);
                },
                None => {
                    return samples;
                }
            }
        }
        // return the full list of samples, removing the final sample
        // if adding 2 samples took us over the max
        if samples.len() > max_samples {
            samples.pop();
            return samples;
        }
        samples
    }

    fn generate_pair<I>(&self, source: &mut I) -> Option<(T, T)>
    where I: Iterator<Item = T> {
        Some((source.next()?, source.next()?))
    }
}
