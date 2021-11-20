extern crate rand;
extern crate rand_chacha;

extern crate textplots;

extern crate easy_ml;

#[cfg(test)]
mod tests {
    use rand::distributions::Standard;
    use rand::{Rng, SeedableRng};

    use textplots::{Chart, Plot, Shape};

    use easy_ml::distributions::{Gaussian, MultivariateGaussian};
    use easy_ml::matrices::slices::{Slice, Slice2D};
    use easy_ml::matrices::Matrix;

    // 3 steps for bayesian regression
    // 0: have data to model
    // 1: create a model from a prior distribution
    // 2: observe data
    // 3: update beliefs

    const LINES_TO_DRAW: usize = 5;
    const SAMPLES_FOR_DISTRIBUTION: usize = 500;

    /**
     * The function of x without any noise
     */
    fn target(x: f64) -> f64 {
        (x * 1.5) + 3.0
    }

    fn generate_data<R: Rng>(
        random_generator: &mut R,
        variance: f64,
        training_size: usize,
    ) -> (
        Matrix<f64>,
        Matrix<f64>,
        Matrix<f64>,
        Matrix<f64>,
        Matrix<f64>,
        Matrix<f64>,
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
                .collect(),
        );

        // create noisy target values to train on from these observations
        let normal_distribution = Gaussian::new(0.0, variance);
        let mut random_numbers = n_random_numbers(random_generator, training_size);
        let samples = normal_distribution
            .draw(&mut random_numbers.drain(..), training_size)
            .unwrap();
        let targets = observations.map_with_index(|x, row, _| target(x) + samples[row]);

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

        (
            x,
            y,
            test_design_matrix,
            design_matrix,
            targets,
            observations,
        )
    }

    fn n_random_numbers<R: Rng>(random_generator: &mut R, n: usize) -> Vec<f64> {
        random_generator.sample_iter(Standard).take(n).collect()
    }

    fn merge_for_plotting(x: &Matrix<f64>, fx: &Matrix<f64>) -> Vec<(f32, f32)> {
        x.column_iter(0)
            .zip(fx.column_iter(0))
            .map(|(x, y)| (x as f32, y as f32))
            .collect()
    }

    fn sort_and_merge_for_plotting(x: &Matrix<f64>, fx: &Matrix<f64>) -> Vec<(f32, f32)> {
        let mut list: Vec<(f32, f32)> = x
            .column_iter(0)
            .zip(fx.column_iter(0))
            .map(|(x, y)| (x as f32, y as f32))
            .collect();
        list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        list
    }

    fn split_for_plotting(pairs: &Matrix<f64>) -> Vec<(f32, f32)> {
        pairs
            .column_iter(0)
            .zip(pairs.column_iter(1))
            .map(|(x, y)| (x as f32, y as f32))
            .collect()
    }

    // model is w0 + w1*x = y = f(x, w) where w will be a 2x1 column vector of w0 and w1
    // hence we need basis functions that transform each row of data [x] into [1, x]
    // so when we take the dot product of x and w we compute w0*1 + w1*x = y

    #[test]
    fn test_bayesian_regression() {
        // use a fixed seed random generator from the rand crate
        let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(10);

        // Start by defining the precision (β) (the inverse of variance (σ^2)) of the gaussian
        // noise in the data, in a real world scenario we cannot know this a priori but could still
        // approximate it from the data available.
        // In this example we define β and then compute the noisy line according to
        // the value we choose for simplicity.
        let noise_precision = 25.0;
        let noise_variance = 1.0 / noise_precision;
        let (x, y, test_design_matrix, design_matrix, targets, observations) =
            generate_data(&mut random_generator, noise_variance, 20);

        // plot x and y to see the true line and the approximate line from
        // the noisy data
        println!("True x and f(x) and noisy version");
        Chart::new(180, 60, -1.0, 1.0)
            .lineplot(Shape::Lines(&merge_for_plotting(&x, &y)))
            .lineplot(Shape::Points(&sort_and_merge_for_plotting(
                &observations,
                &targets,
            )))
            .display();

        // Start with a prior distribution which we will update as we see new data.
        // We use 0 mean and the scaled identity matrix as the covariance prior which regularises
        // the bayesian regression towards low values for w0 and w1. To get this regularisation
        // in standard linear regression we would have to add a regularisation parameter, but
        // because bayesian regression starts with a prior we can regularise with the prior.
        // If we thought the line had an intercept around 100 rather than 3 then we could start
        // the prior there to help with the learning.
        // Note also that the prior and posterior distributions are over the parameters
        // of our line w0 + w1*x. They are not distributions over the values of x or f(x).
        // Both the prior and posterior distributions are generative models that we can draw
        // samples for the weights w0 and w1 to then estimate f(x) from x.
        let prior_precision = 1.0;
        let prior_variance = 1.0 / prior_precision;
        let prior = MultivariateGaussian::new(
            Matrix::column(vec![0.0, 0.0]),
            Matrix::diagonal(prior_variance, (2, 2)),
        );

        // Draw some lines from the prior before seeing data to see what our
        // prior belief looks like
        let mut random_numbers = n_random_numbers(&mut random_generator, LINES_TO_DRAW * 2);
        let weights = prior
            .draw(&mut random_numbers.drain(..), LINES_TO_DRAW)
            .unwrap();
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
                &Matrix::column(predicted_targets.column_iter(i).collect()),
            )));
        }
        chart.display();

        // draw more weights to plot the distribution of weights in the prior
        println!("Weights distribution of prior (w1 and w0)");
        let mut random_numbers =
            n_random_numbers(&mut random_generator, SAMPLES_FOR_DISTRIBUTION * 2);
        let weights = prior
            .draw(&mut random_numbers.drain(..), SAMPLES_FOR_DISTRIBUTION)
            .unwrap();
        let mut chart = Chart::new(80, 80, -3.0, 3.0);
        chart.lineplot(Shape::Points(&split_for_plotting(&weights)));
        chart.display();

        for training_size in vec![1, 2, 5, 20] {
            println!("Training size: {}", training_size);

            // use increasing amounts of training samples to see the effect
            // on the posterior as more evidence is seen
            let design_matrix_n = design_matrix.retain(
                Slice2D::new()
                    .rows(Slice::Range(0..training_size))
                    .columns(Slice::All()),
            );

            let targets_n = targets.retain(
                Slice2D::new()
                    .rows(Slice::Range(0..training_size))
                    .columns(Slice::All()),
            );

            let observations_n = observations.retain(
                Slice2D::new()
                    .rows(Slice::Range(0..training_size))
                    .columns(Slice::All()),
            );

            println!("Observations for N={}", training_size);
            Chart::new(180, 60, -1.0, 1.0)
                .lineplot(Shape::Points(&sort_and_merge_for_plotting(
                    &observations_n,
                    &targets_n,
                )))
                .display();

            // General case for multivariate regression is
            // Prior is N(u_prior, C_prior)
            // (C_n)^-1 is (X^T * C_error^-1 * X) + (C_prior)^-1
            // u_n is C_n * ((X^T * C_error * targets_n) + (((C_prior)^-1) * u_prior))

            let new_precision = Matrix::diagonal(prior_precision, (2, 2))
                + (design_matrix_n.transpose() * &design_matrix_n).map(|x| x * noise_precision);
            let new_covariance = new_precision.inverse().unwrap();
            let new_mean = new_covariance.map(|x| x * noise_precision)
                * design_matrix_n.transpose()
                * &targets_n;

            let posterior = MultivariateGaussian::new(new_mean, new_covariance);

            // then draw a few samples from the new Gaussian having seen N data
            // and use these w0 and w1 parameters drawn to create a few lines
            // draw MxN random numbers because N is even
            let mut random_numbers = n_random_numbers(&mut random_generator, LINES_TO_DRAW * 2);
            let weights = posterior
                .draw(&mut random_numbers.drain(..), LINES_TO_DRAW)
                .unwrap();
            let predicted_targets = &test_design_matrix * weights.transpose();

            // plot the x and predicted to see the lines drawn from the posterior
            // over the whole data range
            println!(
                "True x and f(x) and 5 lines of the parameters drawn from the posterior of N={}",
                training_size
            );
            let mut chart = Chart::new(180, 60, -1.0, 1.0);
            chart.lineplot(Shape::Lines(&merge_for_plotting(&x, &y)));
            for i in 0..LINES_TO_DRAW {
                // slice into each column of the predicted_targets matrix to
                // get the predictions for each set of paramters drawn from the posterior
                chart.lineplot(Shape::Lines(&merge_for_plotting(
                    &x,
                    &Matrix::column(predicted_targets.column_iter(i).collect()),
                )));
            }
            chart.display();

            // draw more weights to plot the distribution of weights in the posterior
            println!(
                "Weights distribution of posterior (w1 and w0) of N={}",
                training_size
            );
            let mut random_numbers =
                n_random_numbers(&mut random_generator, SAMPLES_FOR_DISTRIBUTION * 2);
            let weights = posterior
                .draw(&mut random_numbers.drain(..), SAMPLES_FOR_DISTRIBUTION)
                .unwrap();
            let mut chart = Chart::new(80, 80, 2.0, 4.0);
            chart.lineplot(Shape::Points(&split_for_plotting(&weights)));
            chart.display();

            // From inspecting the distributions of the final N=20 posterior we
            // can see that the model quickly becomes very certain on the value for the intercept
            // in the weights, but still retains some uncertainty on the gradient

            // TODO: Predictive distribution
        }

        //assert_eq!(1, 2);
    }
    // TODO: add appendix on formulas on how to obtain the noise precision and the prior
    // precision by search rather than cheating and taking them as known
}
