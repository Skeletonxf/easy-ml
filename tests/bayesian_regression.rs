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

    const SAMPLES: usize = 10000;

    // 3 steps for bayesian regression
    // 0: have data to model
    // 1: create a model from a prior distribution
    // 2: observe data
    // 3: update beliefs, go to 2

    /**
     * The function of x without any noise
     */
    fn target(x: f64) -> f64 {
        (x * 1.5) + 3.0
    }

    fn generate_data<R: Rng>(random_generator: &mut R, variance: f64) -> (
        Matrix<f64>,
        Matrix<f64>,
        Matrix<f64>,
        Matrix<f64>
    ) {
        // create data range from 0 to 1
        let x = Matrix::column((0..100).map(|x| x as f64 * 0.01).collect());
        // create true y values without any nooise
        let y_true = x.map(|x| target(x));
        // create noisy y values to train on
        let normal_distribution = Gaussian::new(0.0, variance);
        let mut random_numbers = n_random_numbers(random_generator, 100);
        let samples = normal_distribution.draw(&mut random_numbers.drain(..), 100);
        let y = x.map_with_index(|x, row, _| target(x) + samples[row]);

        // create a design matrix of [1, x] for each x
        let X = {
            let mut X = x.clone();
            X.insert_column(0, 1.0);
            X
        };
        (x, X, y, y_true)
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

    #[test]
    fn test_bayesian_regression() {
        // use a fixed seed non cryptographically secure random
        // generator from the rand crate
        let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(10);

        let variance = 1.0 / 25.0;
        let (x, X, y, y_true) = generate_data(&mut random_generator, variance);

        // y_true is a straight line, y is noisy and y_true is also
        // the 'best fit' for y.
        Chart::new(180, 60, 0.0, 1.0)
            .lineplot(Shape::Lines(&merge_for_plotting(&x, &y)))
            .lineplot(Shape::Lines(&merge_for_plotting(&x, &y_true)))
            .nice();

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
        //assert_eq!(1, 2);
    }
}
