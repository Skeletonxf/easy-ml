extern crate easy_ml;

#[cfg(test)]
mod distributions {
    use easy_ml::distributions::{Gaussian, MultivariateGaussian, MultivariateGaussianTensor};
    use easy_ml::matrices::Matrix;
    use easy_ml::tensors::Tensor;

    #[test]
    fn test_normal_distribution() {
        let function: Gaussian<f64> = Gaussian {
            mean: 1.0,
            variance: 2.0,
        };

        let standard_deviation = function.variance.sqrt();

        // test drawing samples
        // for reproducibility we use a fixed source of randomness
        let random_source = vec![
            0.9596857464377826,
            0.013896580086782295,
            0.132219176728827,
            0.2668376444358789,
            0.9485046375910184,
            0.44310734045833367,
            0.035140102381720606,
            0.5191297519269942,
            0.9010694523883847,
            0.20117358201604363,
            0.6579461681015037,
            0.014173578615730431,
        ];
        let max_samples = random_source.len();
        let samples = function
            .draw(&mut random_source.into_iter(), max_samples)
            .unwrap();

        // the mean of the drawn samples should be very close to 1.0
        let mean: f64 = samples.iter().cloned().sum::<f64>() / (max_samples as f64);
        // check the mean of the samples are within 1 standard deviation
        // of the expected mean, as for such a few number of samples we
        // can't expect to get exactly the mean
        assert!(mean < (function.mean + (standard_deviation * 0.5)));
        assert!(mean > (function.mean - (standard_deviation * 0.5)));
    }

    #[test]
    fn test_multivariate_distribution() {
        let mean = Matrix::column(vec![ 0.0, 10.0, -10.0 ]);
        let covariance = Matrix::from(vec![
            vec![  1.0, 0.5, -0.1 ],
            vec![  0.5, 2.0,  0.9 ],
            vec![ -0.1, 0.9,  1.0 ]
        ]);

        let function: MultivariateGaussian<f64> = MultivariateGaussian::new(
            mean,
            covariance
        );

        // test drawing samples
        // for reproducibility we use a fixed source of randomness
        let random_source = vec![
            0.8040186166230938,
            0.580253222290779,
            0.3807224296769691,
            0.21493968506238081,
            0.7492549315762678,
            0.3307385278792596,
            0.7428234020730242,
            0.924520495434979,
            0.7706864587077074,
            0.9606610369139641,
            0.22464359232335274,
            0.7572331492368785,
            0.42525179149566283,
            0.669874551931497,
            0.4900294220194159,
            0.8419224030915755,
            0.2152677317521281,
            0.9047234091130423,
            0.5552287542435432,
            0.0845469814310511,
            0.41907285936702277,
            0.20341459428573838,
            0.348744633661872,
            0.5312939758141078,
            0.2277672193216529,
            0.1688203334261118,
            0.8382370210884449,
            0.019565698056365877,
            0.6201664569519008,
            0.7479491404823113,
            0.484013003448045,
            0.7347758377654283,
            0.336498328409339,
            0.9071544418215385,
            0.6634427419789561,
            0.5844436681111158,
            0.003789929995079211,
            0.15610119074707995,
            0.2150253132088742,
            0.45046274535042086,
            0.2120243842318259,
            0.9372863373271367,
            0.30736703756192063,
            0.3679450747723736,
            0.6542627475645721,
            0.46319586776390764,
            0.8711072275933169,
            0.2902645933689898,
            0.9720242533821568,
            0.8461652295777833,
            0.543197353963871,
            0.170283015830482,
            0.04273291417037717,
            0.7186183438469571,
            0.6060819803673965,
            0.92429551494178,
            0.9596189866410694,
            0.9415763505896844,
            0.264649901252882,
            0.6987701655198049,
            0.17563937503343774,
            0.46796285074389776,
            0.31485784595214716,
            0.6719786444284983,
            0.10138451740090293,
            0.4985694635269784,
            0.3525591176422236,
            0.3939537113644429,
            0.045029903348206446,
            0.15755561321672773,
            0.254414485279737,
            0.7848580636066318,
            0.05721098529487523,
            0.7601928733446881,
            0.6764204954104394,
            0.7442849656738886,
            0.28162126043898295,
            0.21093997385733432,
            0.9399064250864566,
            0.7977861215591273,
            0.5374688753828565,
            0.8851215426025814,
            0.41642367914454037,
            0.2959207016044503,
            0.2121000397454158,
            0.4868594413558851,
            0.4571656286244179,
            0.13484646900784636,
            0.4443762480787943,
            0.7939780466365716,
            0.7067786522378445,
            0.6152187449980677,
            0.4519149484716358,
            0.7900479010107251,
            0.9689655611261592,
            0.8700246722278424,
            0.3335030618321242,
            0.20847389033467123,
            0.7034874950351062,
            0.8874968748931777
        ];
        let random_source_2 = random_source.clone();

        let max_samples = random_source.len() / 4; // N is 3 so we have to round to 4
        let samples = function
            .draw(&mut random_source.into_iter(), max_samples)
            .unwrap();

        let mean = &function.mean;
        let covariance = &function.covariance;

        // the mean of the drawn samples should be very close to our mean vector
        // check the mean of the samples are within 1 standard deviation
        // of the expected mean, as for such a few number of samples we
        // can't expect to get exactly the mean
        let max_samples = max_samples as f64;

        let feature_1_mean = samples.column_iter(0).sum::<f64>() / max_samples;
        let feature_1_standard_deviation = covariance.get(0, 0).sqrt();
        assert!(feature_1_mean < (mean.get(0, 0) + (feature_1_standard_deviation * 0.5)));
        assert!(feature_1_mean > (mean.get(0, 0) - (feature_1_standard_deviation * 0.5)));

        let feature_2_mean = samples.column_iter(1).sum::<f64>() / max_samples;
        let feature_2_standard_deviation = covariance.get(1, 1).sqrt();
        assert!(feature_2_mean < (mean.get(1, 0) + (feature_2_standard_deviation * 0.5)));
        assert!(feature_2_mean > (mean.get(1, 0) - (feature_2_standard_deviation * 0.5)));

        let feature_3_mean = samples.column_iter(2).sum::<f64>() / max_samples;
        let feature_3_standard_deviation = covariance.get(2, 2).sqrt();
        assert!(feature_3_mean < (mean.get(2, 0) + (feature_3_standard_deviation * 0.5)));
        assert!(feature_3_mean > (mean.get(2, 0) - (feature_3_standard_deviation * 0.5)));

        // Not really sure how to test correlation?

        // Verify tensor version produces the same outputs if we give it the same inputs
        // (since algorithm should be identical)
        let mean = Tensor::from([("means", 3)], vec![ 0.0, 10.0, -10.0 ]);
        let covariance = Tensor::from([("rows", 3), ("columns", 3)], vec![
            1.0, 0.5, -0.1,
            0.5, 2.0,  0.9,
            -0.1, 0.9,  1.0
        ]);

        let function: MultivariateGaussianTensor<f64> = MultivariateGaussianTensor::new(
            mean,
            covariance
        ).unwrap();

        let samples_tensor = function
            .draw(
                &mut random_source_2.into_iter(),
                max_samples as usize,
                "samples",
                "features"
            )
            .unwrap();
        let samples_tensor = samples_tensor.into_matrix();
        assert_eq!(samples, samples_tensor);
    }
}
