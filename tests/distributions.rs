extern crate easy_ml;

#[cfg(test)]
mod tests {
    use easy_ml::distributions::Gaussian;

    #[test]
    fn test_normal_distribution() {
        let function: Gaussian<f64> = Gaussian {
            mean: 1.0,
            variance: 2.0,
        };

        let standard_deviation = function.variance.sqrt();

        // test drawing samples
        // for reproducibility we use a fixed source of randomness
        let mut random_source = vec![
            0.9596857464377826, 0.013896580086782295,
            0.132219176728827, 0.2668376444358789,
            0.9485046375910184, 0.44310734045833367,
            0.035140102381720606, 0.5191297519269942,
            0.9010694523883847, 0.20117358201604363,
            0.6579461681015037, 0.014173578615730431
        ];
        let max_samples = random_source.len();
        let samples = function.draw(&mut random_source.drain(..), max_samples);

        // the mean of the drawn samples should be very close to 1.0
        let mean: f64 = samples.iter().cloned().sum::<f64>() / (max_samples as f64);
        // check the mean of the samples are within 1 standard deviation
        // of the expected mean, as for such a few number of samples we
        // can't expect to get exactly the mean
        assert!(mean < (function.mean + (standard_deviation * 0.5)));
        assert!(mean > (function.mean - (standard_deviation * 0.5)));
    }
}
