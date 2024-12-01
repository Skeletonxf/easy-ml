/*!
Models of distributions that samples can be drawn from.

These structs and methods require numerical types that can be treated as real numbers, ie
unsigned and signed integers cannot be used here.

# Example of plotting a Gaussian

```
extern crate rand;
extern crate rand_chacha;
extern crate textplots;
extern crate easy_ml;

use rand::{Rng, SeedableRng};
use rand::distributions::{DistIter, Standard};
use rand_chacha::ChaCha8Rng;
use textplots::{Chart, Plot, Shape};
use easy_ml::distributions::Gaussian;

const SAMPLES: usize = 10000;

// create a normal distribution, note that the mean and variance are
// given in floating point notation as this will be a f64 Gaussian
let normal_distribution = Gaussian::new(0.0, 1.0);

// first create random numbers between 0 and 1
// using a fixed seed random generator from the rand crate
let mut random_generator = ChaCha8Rng::seed_from_u64(10);
let mut random_numbers: DistIter<Standard, &mut ChaCha8Rng, f64> =
    (&mut random_generator).sample_iter(Standard);

// draw samples from the normal distribution
let samples: Vec<f64> = normal_distribution.draw(&mut random_numbers, SAMPLES)
    // unwrap is perfectly safe if and only if we know we have supplied enough random numbers
    .unwrap();

// create a [(f32, f32)] list to plot a histogram of
let histogram_points = {
    let x = 0..SAMPLES;
    let mut y = samples;
    let mut points = Vec::with_capacity(SAMPLES);
    for (x, y) in y.into_iter().zip(x).map(|(y, x)| (x as f32, y as f32)) {
        points.push((x, y));
    }
    points
};

// Plot a histogram from -3 to 3 with 30 bins to check that this distribution
// looks like a Gaussian. This will show a bell curve for large enough SAMPLES.
let histogram = textplots::utils::histogram(&histogram_points, -3.0, 3.0, 30);
Chart::new(180, 60, -3.0, 3.0)
    .lineplot(&Shape::Bars(&histogram))
    .nice();
```

# Getting an infinite iterator using the rand crate

It may be convenient to create an infinite iterator for random numbers so you don't need
to populate lists of random numbers when using these types.

```
use rand::{Rng, SeedableRng};
use rand::distributions::{DistIter, Standard};
use rand_chacha::ChaCha8Rng;

// using a fixed seed random generator from the rand crate
let mut random_generator = ChaCha8Rng::seed_from_u64(16);
// now pass this Iterator to Gaussian functions that accept a &mut Iterator
let mut random_numbers: DistIter<Standard, &mut ChaCha8Rng, f64> =
    (&mut random_generator).sample_iter(Standard);
```

# Example of creating an infinite iterator

The below example is for reference, don't actually do this if you're using rand because rand
can give you an infinite iterator already (see above example).

```
use rand::{Rng, SeedableRng};

// using a fixed seed random generator from the rand crate
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

// now pass this Iterator to Gaussian functions that accept a &mut Iterator
let mut random_numbers = EndlessRandomGenerator { rng: random_generator };
```

# Example of creating an infinite iterator for web assembly targets
[See web_assembly module for Example of creating an infinite iterator for web assembly targets](super::web_assembly)
 */

use crate::numeric::extra::{Real, RealRef};
use crate::linear_algebra;
use crate::matrices::Matrix;
use crate::tensors::views::{TensorRef, TensorView};
use crate::tensors::{Dimension, Tensor};

use std::error::Error;
use std::fmt;

/**
 * A Gaussian probability density function of a normally distributed
 * random variable with expected value / mean μ, and variance σ<sup>2</sup>.
 *
 * See: [https://en.wikipedia.org/wiki/Gaussian_function](https://en.wikipedia.org/wiki/Gaussian_function)
 */
#[derive(Clone, Debug)]
pub struct Gaussian<T: Real> {
    /**
     * The mean is the expected value of this gaussian.
     */
    pub mean: T,
    /**
     * The variance is a measure of the spread of values around the mean, high variance means
     * one standard deviation encompasses a larger spread of values from the mean.
     */
    pub variance: T,
}

impl<T: Real> Gaussian<T> {
    pub fn new(mean: T, variance: T) -> Gaussian<T> {
        Gaussian { mean, variance }
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
    where
        I: Iterator<Item = T>,
    {
        let mut copy: Vec<T> = data.collect();
        Gaussian {
            // duplicate the data to pass once each to mean and variance
            // functions of linear_algebra
            mean: linear_algebra::mean(copy.iter().cloned()),
            variance: linear_algebra::variance(copy.drain(..)),
        }
    }
}

impl<T: Real> Gaussian<T>
where
    for<'a> &'a T: RealRef<T>,
{
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
        let exponent = (-T::one() / &two) * ((x - &self.mean) / &self.variance).pow(&two);
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
     * [Example of generating and feeding random numbers](self)
     */
    pub fn draw<I>(&self, source: &mut I, max_samples: usize) -> Option<Vec<T>>
    where
        I: Iterator<Item = T>,
    {
        let two = T::one() + T::one();
        let minus_two = -&two;
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
    where
        I: Iterator<Item = T>,
    {
        Some((source.next()?, source.next()?))
    }

    #[deprecated(since = "1.1.0", note = "renamed to `probability`")]
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
 * The mean [Matrix] must always be a column vector, and must be the same length as the
 * covariance matrix.
 */
#[derive(Clone, Debug)]
pub struct MultivariateGaussian<T: Real> {
    mean: Matrix<T>,
    covariance: Matrix<T>,
}

impl<T: Real> MultivariateGaussian<T> {
    /**
     * Constructs a new multivariate Gaussian distribution from
     * a Nx1 column vector of means and a NxN covariance matrix
     *
     * This function does not check that the provided covariance matrix
     * is actually a covariance matrix. If a square matrix that is not
     * symmetric is supplied the Gaussian is not defined.
     *
     * # Panics
     *
     * Panics if the covariance matrix is not square, or the column vector
     * is not the same length as the covariance matrix size. Does not currently
     * panic if the covariance matrix is not symmetric, but this could be checked
     * in the future.
     */
    pub fn new(mean: Matrix<T>, covariance: Matrix<T>) -> MultivariateGaussian<T> {
        assert!(mean.columns() == 1, "Mean must be a column vector");
        assert!(
            covariance.rows() == covariance.columns(),
            "Supplied 'covariance' matrix is not square"
        );
        assert!(
            mean.rows() == covariance.rows(),
            "Means must be same length as covariance matrix"
        );
        MultivariateGaussian { mean, covariance }
    }

    /**
     * The mean is a column vector of expected values in each dimension
     */
    pub fn mean(&self) -> &Matrix<T> {
        &self.mean
    }

    /**
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
    pub fn covariance(&self) -> &Matrix<T> {
        &self.covariance
    }
}

impl<T: Real> MultivariateGaussian<T>
where
    for<'a> &'a T: RealRef<T>,
{
    /**
     * Draws samples from this multivariate distribution, provided that the covariance
     * matrix is positive definite.
     *
     * For max_samples of M, sufficient random numbers from the source iterator in the uniformly
     * distributed range [0, 1] inclusive, and this Gaussian's dimensionality of N, returns an
     * MxN matrix of drawn values.
     *
     * The source iterator must have at least MxN random values if N is even, and
     * Mx(N+1) random values if N is odd, or `None` will be returned.
     *
     * [Example of generating and feeding random numbers](super::k_means)
     *
     * If the covariance matrix is only positive semi definite, `None` is returned. You
     * can check if a given covariance matrix is positive definite instead of just positive semi
     * definite with the [cholesky](linear_algebra::cholesky_decomposition) decomposition.
     */
    pub fn draw<I>(&self, source: &mut I, max_samples: usize) -> Option<Matrix<T>>
    where
        I: Iterator<Item = T>,
    {
        use crate::interop::{DimensionNames, RowAndColumn, TensorRefMatrix};
        // Since we already validated our state on construction, we wouldn't expect these
        // conversions to fail but if they do return None
        // Convert the column vector to a 1 dimensional tensor by selecting the sole column
        let mean = crate::tensors::views::TensorIndex::from(
            TensorRefMatrix::from(&self.mean).ok()?,
            [(RowAndColumn.names()[1], 0)],
        );
        let covariance = TensorRefMatrix::from(&self.covariance).ok()?;
        let samples = draw_tensor_samples::<T, _, _, _>(
            &mean,
            &covariance,
            source,
            max_samples,
            "samples",
            "features",
        );
        samples.map(|tensor| tensor.into_matrix())
    }
}

/**
 * A multivariate Gaussian distribution with mean vector μ, and covariance matrix Σ.
 *
 * See: [https://en.wikipedia.org/wiki/Multivariate_normal_distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
 */
#[derive(Clone, Debug)]
pub struct MultivariateGaussianTensor<T: Real> {
    mean: Tensor<T, 1>,
    covariance: Tensor<T, 2>,
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum MultivariateGaussianError<T> {
    NotCovarianceMatrix {
        mean: Tensor<T, 1>,
        covariance: Tensor<T, 2>,
    },
    MeanVectorWrongLength {
        mean: Tensor<T, 1>,
        covariance: Tensor<T, 2>,
    },
}

impl<T: fmt::Debug> fmt::Display for MultivariateGaussianError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultivariateGaussianError::NotCovarianceMatrix {
                mean: _,
                covariance,
            } => write!(f, "Covariance matrix is not square: {:?}", covariance,),
            MultivariateGaussianError::MeanVectorWrongLength { mean, covariance } => write!(
                f,
                "Mean vector has a different length {:?} to the covariance matrix size: {:?}",
                mean.shape(),
                covariance.shape(),
            ),
        }
    }
}

impl<T: fmt::Debug> Error for MultivariateGaussianError<T> {}

impl<T: Real> MultivariateGaussianTensor<T> {
    /**
     * Constructs a new multivariate Gaussian distribution from
     * a N length vector of means and a NxN covariance matrix
     *
     * This function does not check that the provided covariance matrix
     * is actually a covariance matrix. If a square matrix that is not
     * symmetric is supplied the Gaussian is not defined.
     *
     * Result::Err is returned if the covariance matrix is not square, or the mean
     * vector is not the same length as the size of the covariance matrix. Does not currently
     * panic if the covariance matrix is not symmetric, but this could be checked
     * in the future.
     *
     * The dimension names of the mean and covariance matrix are not used, and do not need
     * to match.
     */
    // Boxing error variant as per clippy lint that MultivariateGaussianError is 152 bytes which
    // is kinda big
    pub fn new(
        mean: Tensor<T, 1>,
        covariance: Tensor<T, 2>,
    ) -> Result<MultivariateGaussianTensor<T>, Box<MultivariateGaussianError<T>>> {
        let covariance_shape = covariance.shape();
        if !crate::tensors::dimensions::is_square(&covariance_shape) {
            return Err(Box::new(MultivariateGaussianError::NotCovarianceMatrix {
                mean,
                covariance,
            }));
        }
        let length = covariance_shape[0].1;
        if mean.shape()[0].1 != length {
            return Err(Box::new(MultivariateGaussianError::MeanVectorWrongLength {
                mean,
                covariance,
            }));
        }
        Ok(MultivariateGaussianTensor { mean, covariance })
    }

    /**
     * The mean is a vector of expected values in each dimension
     */
    pub fn mean(&self) -> &Tensor<T, 1> {
        &self.mean
    }

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
    pub fn covariance(&self) -> &Tensor<T, 2> {
        &self.covariance
    }
}

fn draw_tensor_samples<T, S1, S2, I>(
    mean: S1,
    covariance: S2,
    source: &mut I,
    max_samples: usize,
    samples: Dimension,
    features: Dimension,
) -> Option<Tensor<T, 2>>
where
    T: Real,
    for<'a> &'a T: RealRef<T>,
    I: Iterator<Item = T>,
    S1: TensorRef<T, 1>,
    S2: TensorRef<T, 2>,
{
    if samples == features {
        return None;
    }
    let mean = TensorView::from(mean);
    let covariance = TensorView::from(covariance);
    use linear_algebra::cholesky_decomposition_tensor as cholesky_decomposition;
    // Follow the method outlined at
    // https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Computational_methods
    let normal_distribution = Gaussian::new(T::zero(), T::one());
    let mut lower_triangular = cholesky_decomposition::<T, _, _>(&covariance)?;
    lower_triangular.rename([samples, features]);

    let number_of_samples = max_samples;
    let number_of_features = mean.shape()[0].1;
    let shape = [(samples, max_samples), (features, number_of_features)];
    let mut drawn_samples = Tensor::empty(shape, T::zero());

    // Construct a TensorView from the mean vector with a shape of
    // [(samples, number_of_features), (features, 1)]
    let column_vector_mean = mean.rename_view([samples]).expand_owned([(1, features)]);

    let mut drawn_samples_iterator = drawn_samples.iter_reference_mut();
    for _sample_row in 0..number_of_samples {
        // use the box muller transform to get N independent values from
        // a normal distribution (x)
        let standard_normals = normal_distribution.draw(source, number_of_features)?;
        let standard_normals = Tensor::from(
            // Construct a column vector with as many rows as our N features
            [(samples, standard_normals.len()), (features, 1)],
            standard_normals,
        );
        // mean + (L * standard_normals) yields each m'th vector from the distribution
        let random_vector = &column_vector_mean + (&lower_triangular * standard_normals);
        // We now have an Nx1 matrix of samples which we can assign to this sample row vector
        for x in random_vector.iter() {
            // Since we'll assign a value exactly number_of_samples * number_of_features times
            // this will always be the Some case
            *drawn_samples_iterator.next()? = x;
        }
    }
    Some(drawn_samples)
}

impl<T: Real> MultivariateGaussianTensor<T>
where
    for<'a> &'a T: RealRef<T>,
{
    /**
     * Draws samples from this multivariate distribution, provided that the covariance
     * matrix is positive definite.
     *
     * For max_samples of M, sufficient random numbers from the source iterator, in the uniformly
     * distributed range [0, 1] inclusive and this Gaussian's dimensionality of N, returns an MxN
     * matrix of drawn values with dimension names `samples` and `features` for M and N respectively.
     *
     * The source iterator must have at least MxN random values if N is even, and
     * Mx(N+1) random values if N is odd, or `None` will be returned.
     *
     * [Example of generating and feeding random numbers](super::k_means)
     *
     * If the covariance matrix is only positive semi definite, `None` is returned. You
     * can check if a given covariance matrix is positive definite instead of just positive semi
     * definite with the [cholesky](linear_algebra::cholesky_decomposition_tensor) decomposition.
     */
    pub fn draw<I>(
        &self,
        source: &mut I,
        max_samples: usize,
        samples: Dimension,
        features: Dimension,
    ) -> Option<Tensor<T, 2>>
    where
        I: Iterator<Item = T>,
    {
        draw_tensor_samples::<T, _, _, _>(
            &self.mean,
            &self.covariance,
            source,
            max_samples,
            samples,
            features,
        )
    }
}
