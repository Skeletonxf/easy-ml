extern crate rand;
extern crate rand_chacha;

extern crate textplots;

extern crate easy_ml;

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use textplots::{Chart, Plot, Shape};

    use easy_ml::distributions::{Gaussian, MultivariateGaussian};
    use easy_ml::matrices::Matrix;
    use easy_ml::matrices::slices::Slice;

    // 3 steps for bayesian regression
    // 0: have data to model
    // 1: create a model from a prior distribution
    // 2: observe data
    // 3: update beliefs

    const LINES_TO_DRAW: usize = 5;

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
        let design_matrix = {
            let mut design_matrix = observations.clone();
            design_matrix.insert_column(0, 1.0);
            design_matrix
        };

        let test_design_matrix = {
            let mut test_design_matrix = x.clone();
            test_design_matrix.insert_column(0, 1.0);
            test_design_matrix
        };

        (x, y, test_design_matrix, design_matrix, targets, observations)
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

        let noise_precision = 25.0;
        let noise_variance = 1.0 / noise_precision;
        let (x, y, test_design_matrix, design_matrix, targets, observations) = generate_data(
            &mut random_generator, noise_variance, 20);

        // plot x and y to see the true line and the approximate line from
        // the noisy data
        println!("True x and f(x) and noisy version");
        Chart::new(180, 60, -1.0, 1.0)
            .lineplot(Shape::Lines(&merge_for_plotting(&x, &y)))
            .lineplot(Shape::Lines(&sort_and_merge_for_plotting(&observations, &targets)))
            .nice();

        // TODO: cover how to obtain the noise precision and the prior precision
        // by search rather than cheating and taking them as known

        let prior_precision = 1.0;
        let prior_variance = 1.0 / prior_precision;

        // start with a prior distribution which we will update as we see new data
        // for simplicity we use 0 mean and the scaled identity matrix as the covariance
        // this is 2 dimensional because we have two paramters w0 and w1 to draw
        // from this distribution

        let prior = MultivariateGaussian::new(
            Matrix::column(vec![ 0.0, 0.0]),
            Matrix::diagonal(prior_variance, (2, 2)));

        let mut random_numbers = n_random_numbers(&mut random_generator, LINES_TO_DRAW * 2);

        // Draw some lines from the prior before seeing data to see what our
        // prior belief looks like
        let weights = prior.draw(&mut random_numbers.drain(..), LINES_TO_DRAW).unwrap();
        let predicted_targets = &test_design_matrix * weights.transpose();

        // plot the x and predicted to see the lines drawn from the posterior
        // over the whole data range
        println!("True x and f(x) and 5 lines of the parameters drawn from the prior");
        let mut chart = Chart::new(180, 60, -1.0, 1.0);
        chart.lineplot(Shape::Lines(&merge_for_plotting(&x, &y)));
        for i in 0..LINES_TO_DRAW {
            // slice into each column of the predicted_targets matrix to
            // get the predictions for each set of paramters drawn from the posterior
            chart.lineplot(Shape::Lines(&merge_for_plotting(
                &x,
                &Matrix::column(predicted_targets.column_iter(i).collect()))));
        }
        chart.nice();

        for training_size in vec![1, 2, 5, 20] {
            println!("Training size: {}", training_size);

            // use increasing amounts of training samples to see the effect
            // on the posterior as more evidence is seen
            let design_matrix_n = design_matrix.retain(Slice::RowRange(0..training_size));

            println!("Observations: {:?}", design_matrix_n);
            let targets = Matrix::column(targets.column_iter(0).take(training_size).collect());
            println!("Targets: {:?}", targets);

            let new_precision = Matrix::diagonal(prior_precision, (2, 2))
                + (design_matrix_n.transpose() * &design_matrix_n).map(|x| x * noise_precision);
            let new_covariance = new_precision.inverse().unwrap();
            println!("Covariance of posterior: {:?}", new_covariance);
            let new_mean = new_covariance.map(|x| x * noise_precision)
                * design_matrix_n.transpose() * &targets;
            println!("Mean of posterior: {:?}", new_mean);

            let posterior = MultivariateGaussian::new(new_mean, new_covariance);

            // then draw a few samples from the new Gaussian having seen N data
            // and use these w0 and w1 parameters drawn to create a few lines
            // draw MxN random numbers because N is even
            let mut random_numbers = n_random_numbers(&mut random_generator, LINES_TO_DRAW * 2);

            let weights = posterior.draw(&mut random_numbers.drain(..), LINES_TO_DRAW).unwrap();
            println!("Data: {}, Weights:\n{:?}", training_size, weights);
            let predicted_targets = &test_design_matrix * weights.transpose();

            // plot the x and predicted to see the lines drawn from the posterior
            // over the whole data range
            println!("True x and f(x) and 5 lines of the parameters drawn from the posterior of N={}", training_size);
            let mut chart = Chart::new(180, 60, -1.0, 1.0);
            chart.lineplot(Shape::Lines(&merge_for_plotting(&x, &y)));
            for i in 0..LINES_TO_DRAW {
                // slice into each column of the predicted_targets matrix to
                // get the predictions for each set of paramters drawn from the posterior
                chart.lineplot(Shape::Lines(&merge_for_plotting(
                    &x,
                    &Matrix::column(predicted_targets.column_iter(i).collect()))));
            }
            chart.nice();

            // then plot the sample of lines and the true one
            // and the datapoints seen
        }

        assert_eq!(1, 2);
    }
}
