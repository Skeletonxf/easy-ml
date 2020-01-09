/*!
Models of distributions that samples can be drawn from.

# Example of plotting a Gaussian

```
extern crate rand;
extern crate rand_chacha;
extern crate textplots;
extern crate easy_ml;

use rand::{Rng, SeedableRng};
use textplots::{Chart, Plot, Shape};
use easy_ml::distributions::Gaussian;

const SAMPLES: usize = 10000;

// create a normal distribution, note the mean and variance are
// given in floating point notation as this will be a f64 Gaussian
let normal_distribution = Gaussian::new(0.0, 1.0);

// first create random numbers between 0 and 1
// using a fixed seed non cryptographically secure random
// generator from the rand crate
let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(10);

let mut random_numbers = Vec::with_capacity(SAMPLES);
for _ in 0..SAMPLES {
    random_numbers.push(random_generator.gen::<f64>());
}

// draw samples from the normal distribution
let samples: Vec<f64> = normal_distribution.draw(&mut random_numbers.drain(..), SAMPLES);

// create a [(f32, f32)] list to plot a histogram of
let histogram_points = {
    let x = 0..SAMPLES;
    let mut y = samples;
    let mut points = Vec::with_capacity(SAMPLES);
    for (x, y) in y.drain(..).zip(x).map(|(y, x)| (x as f32, y as f32)) {
        points.push((x, y));
    }
    points
};

// plot a histogram from -3 to 3 with 30 bins
// to check that this distribution looks like a gaussian
// this will show a bell curve for large enough SAMPLES.
let histogram = textplots::utils::histogram(&histogram_points, -3.0, 3.0, 30);
Chart::new(180, 60, -3.0, 3.0)
    .lineplot( Shape::Bars(&histogram) )
    .nice();
```
 */

use crate::numeric::{Numeric, NumericRef};
use crate::numeric::extra::{Sqrt, Pi, Exp, Pow, Ln, Sin, Cos};
use crate::matrices::Matrix;
use crate::linear_algebra;

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

impl <T: Numeric> Gaussian<T> {
    pub fn new(mean: T, variance: T) -> Gaussian<T> {
        Gaussian {
            mean,
            variance,
        }
    }
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


/**
 * A multivariate Gaussian distribution with mean vector μ, and covariance matrix Σ.
 *
 * See: [https://en.wikipedia.org/wiki/Multivariate_normal_distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
 *
 * The mean [Matrix](./../matrices/struct.Matrix.html) must always be a column vector.
 */
pub struct MultivariateGaussian<T: Numeric> {
    pub mean: Matrix<T>,
    pub covariance: Matrix<T>
}

impl <T: Numeric> MultivariateGaussian<T> {
    /**
     * Constructs a new multivariate Gaussian distribution from
     * a Nx1 column vector of means and a NxN covariance matrix
     */
    pub fn new(mean: Matrix<T>, covariance: Matrix<T>) -> MultivariateGaussian<T> {
        // TODO: check arguments
        MultivariateGaussian {
            mean,
            covariance,
        }
    }
}

impl <T: Numeric> MultivariateGaussian<T>
where for<'a> &'a T: NumericRef<T> {
    /**
     * Draws samples from this multivariate distribution.
     *
     * For max_samples of M and sufficient random numbers from the source iterator,
     * returns an MxN matrix of drawn values.
     *
     * The source iterator must have MxN random values if N is even, and
     * Mx(N+1) random values if N is odd, or `None` will be returned. If
     * the cholesky decomposition cannot be taken on this Gaussian's
     * covariance matrix then `None` is also returned.
     */
    pub fn draw<I>(&self, source: &mut I, max_samples: usize) -> Option<Matrix<T>>
        where
            T: Pi + Sqrt<Output = T> + Cos<Output = T> + Sin<Output = T>,
            I: Iterator<Item = T>,
            for<'a> &'a T: Ln<Output = T>,
            for<'a> &'a T: Sqrt<Output = T>, {
        // Follow the method outlined at
        // https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Computational_methods
        let normal_distribution = Gaussian::new(T::zero(), T::one());

        // hope cholesky works for now and check later
        let lower_triangular = linear_algebra::cholesky_decomposition(&self.covariance)?;

        let mut samples = Matrix::empty(T::zero(), (max_samples, self.mean.rows()));

        for row in 0..samples.rows() {
            // use the box muller transform to get N independent values from
            //  a normal distribution (x)
            let standard_normals = normal_distribution.draw(source, self.mean.rows());
            if standard_normals.len() < self.mean.rows() {
                return None;
            }
            // mean + (L * standard_normals) yields each m'th vector from the distribution
            let random_vector = &self.mean + (&lower_triangular * Matrix::column(standard_normals));
            for x in 0..random_vector.columns() {
                samples.set(row, x, random_vector.get(0, x));
            }
        }
        Some(samples)
    }
}
