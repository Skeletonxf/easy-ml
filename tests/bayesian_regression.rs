#![allow(non_snake_case)]

extern crate rand;
extern crate rand_chacha;

extern crate textplots;

extern crate easy_ml;

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use textplots::{Chart, Plot, Shape};

    use easy_ml::distributions::Gaussian;
    use easy_ml::matrices::Matrix;

    // 3 steps for bayesian regression
    // 0: have data to model
    // 1: create a model from a prior distribution
    // 2: observe data
    // 3: update beliefs

    /**
     * The function of x without any noise
     */
    fn target(x: f64) -> f64 {
        (x * 1.5) + 3.0
    }

    fn generate_data<R: Rng>(random_generator: &mut R, variance: f64, training_size: usize) -> (
        Matrix<f64>,
        Matrix<f64>,
        Matrix<f64>,
        Matrix<f64>,
        Matrix<f64>
    ) {
        // create data range from -1 to 1
        let x = Matrix::column((0..100).map(|x| ((x as f64 * 0.01) * 2.0) - 1.0).collect());
        // create y values without any noise
        let y = x.map(|x| target(x));

        // create some random observations in the -1 to 1 range to train on
        let observations = Matrix::column(
            n_random_numbers(random_generator, training_size)
                .iter()
                .map(|x| (x * 2.0) - 1.0)
                .collect());

        // create noisy target values to train on from these observations
        let normal_distribution = Gaussian::new(0.0, variance);
        let mut random_numbers = n_random_numbers(random_generator, training_size);
        let samples = normal_distribution.draw(&mut random_numbers.drain(..), training_size);
        let targets = observations.map_with_index(
            |x, row, _| target(x) + samples[row]);

        // create a design matrix of [1, x] for each x in observations
        let X = {
            let mut X = observations.clone();
            X.insert_column(0, 1.0);
            X
        };
        (x, y, X, targets, observations)
    }

    fn n_random_numbers<R: Rng>(random_generator: &mut R, n: usize) -> Vec<f64> {
        let mut random_numbers = Vec::with_capacity(n);
        for _ in 0..n {
            random_numbers.push(random_generator.gen::<f64>());
        }
        random_numbers
    }

    fn merge_for_plotting(x: &Matrix<f64>, fx: &Matrix<f64>) -> Vec<(f32, f32)> {
        x.column_iter(0)
            .zip(fx.column_iter(0))
            .map(|(x, y)| (x as f32, y as f32))
            .collect()
    }

    fn sort_and_merge_for_plotting(x: &Matrix<f64>, fx: &Matrix<f64>) -> Vec<(f32, f32)> {
        let mut list: Vec<(f32, f32)> = x.column_iter(0)
            .zip(fx.column_iter(0))
            .map(|(x, y)| (x as f32, y as f32))
            .collect();
        list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        list
    }

    // model is w0 + w1*x = y = f(x, w) where w will be a 2x1 column vector of w0 and w1
    // hence we need basis functions that transform each row of data [x] into [1, x]
    // so when we take the dot product of x and w we compute w0*1 + w1*x = y

    #[test]
    fn test_bayesian_regression() {
        // use a fixed seed non cryptographically secure random
        // generator from the rand crate
        let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(10);

        let variance = 1.0 / 25.0;
        let (x, y, X, targets, observations) = generate_data(&mut random_generator, variance, 20);

        // plot x and y to see the true line and the approximate line from
        // the noisy data
        Chart::new(180, 60, 0.0, 1.0)
            .lineplot(Shape::Lines(&merge_for_plotting(&x, &y)))
            .lineplot(Shape::Lines(&sort_and_merge_for_plotting(&observations, &targets)))
            .nice();

        // start with a prior distribution which we will update as we see new data
        // TODO: this needs to be multivariate to cover w0 and w1
        let prior = Gaussian::new(0.0, 2.0);

        for training_size in vec![1, 2, 5, 20] {
            // use increasing amounts of training samples to see the effect
            // on the posterior as more evidence is seen
            let X = Matrix::column(X.column_iter(0).take(training_size).collect());
            let targets = Matrix::column(targets.column_iter(0).take(training_size).collect());

            // use design matrix X and targets, plus the parameters alpha and beta
            // TODO: work out what alpha and beta mean
            // to compute the posterior Gaussian
            // the posterior needs to be multivariate too

            // then draw a few samples from the new Gaussian having seen N data
            // and use these w0 and w1 parameters drawn to create a few lines

            // then plot the sample of lines and the true one
            // and the datapoints seen
        }

        //
        //
        // let normal_distribution = Gaussian::new(0.0, 1.0);
        //
        //
        //
        // let mut random_numbers = Vec::with_capacity(SAMPLES);
        // for _ in 0..SAMPLES {
        //     random_numbers.push(random_generator.gen::<f64>());
        // }
        //
        // // draw samples from the normal distribution
        // let samples: Vec<f64> = normal_distribution.draw(&mut random_numbers.drain(..), SAMPLES);
        //
        assert_eq!(1, 2);
    }
}
