/*!
Models of distributions that samples can be drawn from.

These structs and methods require numerical types that can be treated as real numbers, ie
unsigned and signed numbers cannot be used here.

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

// create a normal distribution, note that the mean and variance are
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
let samples: Vec<f64> = normal_distribution.draw(&mut random_numbers.drain(..), SAMPLES)
    // unwrap is perfectly save if and only if we know we have supplied enough random numbers
    .unwrap();

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

// Plot a histogram from -3 to 3 with 30 bins to check that this distribution
// looks like a Gaussian. This will show a bell curve for large enough SAMPLES.
let histogram = textplots::utils::histogram(&histogram_points, -3.0, 3.0, 30);
Chart::new(180, 60, -3.0, 3.0)
    .lineplot( Shape::Bars(&histogram) )
    .nice();
```

# Example of creating an infinite iterator using the rand crate

It may be convenient to create an infinite iterator for random numbers so you don't need
to populate lists of random numbers when using these types.

```
use rand::{Rng, SeedableRng};

// use a fixed seed non cryptographically secure random generator from the rand crate
let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(16);

struct EndlessRandomGenerator {
    rng: rand_chacha::ChaCha8Rng
}

impl Iterator for EndlessRandomGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        // always return Some, hence this iterator is infinite
        Some(self.rng.gen::<f64>())
    }
}

// now pass this instance to Gaussian functions that accept a &mut Iterator
let mut random_numbers = EndlessRandomGenerator { rng: random_generator };
```
 */

use crate::numeric::{Numeric, NumericRef};
use crate::numeric::extra::{Real, RealRef};
//use crate::numeric::extra::{Sqrt, Pi, Exp, Pow, Ln, Sin, Cos};
use crate::matrices::Matrix;
use crate::linear_algebra;

/**
 * A Gaussian probability density function of a normally distributed
 * random variable with expected value / mean μ, and variance σ<sup>2</sup>.
 *
 * See: [https://en.wikipedia.org/wiki/Gaussian_function](https://en.wikipedia.org/wiki/Gaussian_function)
 */
#[derive(Debug)]
pub struct Gaussian<T: Numeric + Real> {
    /**
     * The mean is the expected value of this gaussian.
     */
    pub mean: T,
    /**
     * The variance is a measure of the spread of values around the mean, high variance means
     * one standard deviation encompasses a larger spread of values from the mean.
     */
    pub variance: T
}

impl <T: Numeric + Real> Gaussian<T> {
    pub fn new(mean: T, variance: T) -> Gaussian<T> {
        Gaussian {
            mean,
            variance,
        }
    }

    /**
     * Creates a Gaussian approximating the mean and variance in the provided
     * data.
     *
     * Note that this will always be an approximation, if you generate some data
     * according to some mean and variance then construct a Gaussian from
     * the mean and variance of that generated data the approximated mean
     * and variance is unlikely to be exactly the same as the parameters the
     * data was generated with, though as the amout of data increases you
     * can expect the approximation to be more close.
     */
    pub fn approximating<I>(data: I) -> Gaussian<T>
    where I: Iterator<Item = T> {
        let mut copy: Vec<T> = data.collect();
        Gaussian {
            // duplicate the data to pass once each to mean and variance
            // functions of linear_algebra
            mean: linear_algebra::mean(copy.iter().cloned()),
            variance: linear_algebra::variance(copy.drain(..)),
        }
    }
}

impl <T: Numeric + Real> Gaussian<T>
where for<'a> &'a T: NumericRef<T> + RealRef<T> {
    /**
     * Computes g(x) for some x, the probability density of a normally
     * distributed random variable x, or in other words how likely x is
     * to be drawn from this normal distribution.
     *
     * g(x) is largest for x equal to this distribution's mean and
     * g(x) will tend towards zero as x is further from this distribution's
     * mean, at a rate corresponding to this distribution's variance.
     */
    pub fn probability(&self, x: &T) -> T {
        // FIXME: &T sqrt doesn't seem to be picked up by the compiler here
        let standard_deviation = self.variance.clone().sqrt();
        let two = T::one() + T::one();
        let two_pi = &two * T::pi();
        let fraction = T::one() / (standard_deviation * (&two_pi.sqrt()));
        let exponent = (- T::one() / &two) * ((x - &self.mean) / &self.variance).pow(&two);
        fraction * exponent.exp()
    }

    /**
     * Given a source of random variables in the uniformly distributed
     * range [0, 1] inclusive, draws `max_samples` of independent
     * random numbers according to this Gaussian distribution's mean and
     * variance using the Box-Muller transform:
     *
     * [https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
     *
     * The source of random variables must provide at least as many values
     * as `max_samples` if `max_samples` is even, and one more than `max_samples`
     * if `max_samples` is odd. If fewer are provided None is returned.
     *
     * As all randomness is provided to this method, this code is deterministic
     * and will always compute the same samples given the same random source
     * of numbers.
     *
     * [Example of generating and feeding random numbers](./index.html)
     */
    pub fn draw<I>(&self, source: &mut I, max_samples: usize) -> Option<Vec<T>>
    where I: Iterator<Item = T> {
        let two = T::one() + T::one();
        let minus_two = - &two;
        let two_pi = &two * T::pi();
        let mut samples = Vec::with_capacity(max_samples);
        let standard_deviation = self.variance.clone().sqrt();
        // keep drawing samples from this normal Gaussian distribution
        // until either the iterator runs out or we reach the max_samples
        // limit
        while samples.len() < max_samples {
            let (u, v) = self.generate_pair(source)?;
            // these computations convert two samples from the inclusive 0 - 1
            // range to two samples of a normal distribution with with
            // μ = 0 and σ = 1.
            let z1 = (&minus_two * u.clone().ln()).sqrt() * ((&two_pi * &v).cos());
            let z2 = (&minus_two * u.clone().ln()).sqrt() * ((&two_pi * &v).sin());
            // now we scale to the mean and variance for this Gaussian
            let sample1 = (z1 * &standard_deviation) + &self.mean;
            let sample2 = (z2 * &standard_deviation) + &self.mean;
            samples.push(sample1);
            samples.push(sample2);
        }
        // return the full list of samples, removing the final sample
        // if adding 2 samples took us over the max
        if samples.len() > max_samples {
            samples.pop();
            return Some(samples);
        }
        Some(samples)
    }

    fn generate_pair<I>(&self, source: &mut I) -> Option<(T, T)>
    where I: Iterator<Item = T> {
        Some((source.next()?, source.next()?))
    }

    #[deprecated(since="1.1.0", note="renamed to `probability`")]
    pub fn map(&self, x: &T) -> T {
        self.probability(x)
    }
}

/**
 * A multivariate Gaussian distribution with mean vector μ, and covariance matrix Σ.
 *
 * See: [https://en.wikipedia.org/wiki/Multivariate_normal_distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
 *
 * # Invariants
 *
 * The mean [Matrix](./../matrices/struct.Matrix.html) must always be a column vector, and
 * must be the same length as the covariance matrix.
 */
#[derive(Debug)]
pub struct MultivariateGaussian<T: Numeric + Real> {
    /**
     * The mean is a column vector of expected values in each dimension
     */
    pub mean: Matrix<T>,
    /**
     * The covariance matrix is a NxN matrix where N is the number of dimensions for
     * this Gaussian. A covariance matrix must always be symmetric, that is `C[i,j] = C[j,i]`.
     *
     * The covariance matrix is a measure of how much values from each dimension vary
     * from their expected value with respect to each other.
     *
     * For a 2 dimensional multivariate Gaussian the covariance matrix could be the 2x2 identity
     * matrix:
     *
     * ```ignore
     * [
     *   1.0, 0.0
     *   0.0, 1.0
     * ]
     * ```
     *
     * In which case the two dimensions are completely uncorrelated as `C[0,1] = C[1,0] = 0`.
     */
    pub covariance: Matrix<T>
}

impl <T: Numeric + Real> MultivariateGaussian<T> {
    /**
     * Constructs a new multivariate Gaussian distribution from
     * a Nx1 column vector of means and a NxN covariance matrix
     *
     * This function does not check that the provided covariance matrix
     * is actually a covariance matrix. If a square matrix that is not
     * symmetric is supplied the gaussian is not defined.
     */
    pub fn new(mean: Matrix<T>, covariance: Matrix<T>) -> MultivariateGaussian<T> {
        assert!(mean.columns() == 1, "Mean must be a column vector");
        assert!(covariance.rows() == covariance.columns(), "Supplied 'covariance' matrix is not square");
        assert!(mean.rows() == covariance.rows(), "Means must be same length as covariance matrix");
        MultivariateGaussian {
            mean,
            covariance,
        }
    }
}

impl <T: Numeric + Real> MultivariateGaussian<T>
where for<'a> &'a T: NumericRef<T> + RealRef<T> {
    /**
     * Draws samples from this multivariate distribution.
     *
     * For max_samples of M, sufficient random numbers from the source iterator,
     * and this Gaussian's dimensionality of N, returns an MxN matrix of drawn values.
     *
     * The source iterator must have at least MxN random values if N is even, and
     * Mx(N+1) random values if N is odd, or `None` will be returned. If
     * the cholesky decomposition cannot be taken on this Gaussian's
     * covariance matrix then `None` is also returned.
     *
     * [Example of generating and feeding random numbers](../k_means/index.html)
     */
    pub fn draw<I>(&self, source: &mut I, max_samples: usize) -> Option<Matrix<T>>
    where I: Iterator<Item = T> {
        // Follow the method outlined at
        // https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Computational_methods
        let normal_distribution = Gaussian::new(T::zero(), T::one());

        // hope cholesky works for now and check later
        let lower_triangular = linear_algebra::cholesky_decomposition(&self.covariance)?;

        let mut samples = Matrix::empty(T::zero(), (max_samples, self.mean.rows()));

        for row in 0..samples.rows() {
            // use the box muller transform to get N independent values from
            //  a normal distribution (x)
            let standard_normals = normal_distribution.draw(source, self.mean.rows())?;
            // mean + (L * standard_normals) yields each m'th vector from the distribution
            let random_vector = &self.mean + (&lower_triangular * Matrix::column(standard_normals));
            for x in 0..random_vector.rows() {
                samples.set(row, x, random_vector.get(x, 0));
            }
        }
        Some(samples)
    }
}
