/*!
Naïve Bayes examples

# Naïve Bayes

The naïve bayes assumption is that all features in the labelled data are independent of each other
given the class they correspond to. This means the probability of some input given a class
can be computed as the product of each individual feature in that input conditioned on that class.

By Baye's Theorum we can relate the probability of the class given the input to the probability
of the input given the class. As a classifier only needs to determine which class some input
is most likely to be we can compare just the product of the probability of the input given a class
and the probability of that class.

## Bayes' Theorum

posterior = ( prior x likelihood ) / evidence

P(C<sub>k</sub> | **x**) = ( P(C<sub>k</sub>) * P(**x** | C<sub>k</sub>) ) / P(**x**)

P(C<sub>k</sub> | **x**) ∝ P(C<sub>k</sub>) * P(**x** | C<sub>k</sub>)

where C<sub>k</sub> is the kth class and **x** is the input to classify.

## Classifier

taking logs on Bayes' rule yields

log(P(C<sub>k</sub> | **x**)) ∝ log(P(C<sub>k</sub>)) + log(P(**x** | C<sub>k</sub>))

given the naïve bayes assumption

log(P(C<sub>k</sub> | **x**)) ∝ log(P(C<sub>k</sub>)) + the sum over all i features of (log(P(x<sub>i</sub> | C<sub>k</sub>)))

Then to determine the class we take the class corresponding to the largest
log(P(C<sub>k</sub> | **x**)).

Computing the individual probabilities of a feature conditioned on a class depends on what the
type of data is.

For categorical data this is simply occurances in class / total occurances. In practise laplacian
smoothing (adding one occurance for each class to the computed probabilities) may be used to
avoid computing a probability of 0 when some category doesn't have any samples for a class.

For continuous data we can model the feature as distributed according to a Gaussian distribution.

## Simple Naïve Bayes Example with F-1 score analysis

Naïve Bayes can be done by hand (with a calculator) which is what the below example will show.
We have a list of data about the environment and want to predict if we should go outside based
on the conditions. Some of these are categorical values and others are real valued.

```
use easy_ml::distributions::Gaussian;
use easy_ml::linear_algebra;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Weather {
    Stormy, Rainy, Cloudy, Clear, Sunny
}

type WindSpeed = f64;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Pandemic {
    Pandemic, NoPandemic
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Decision {
    GoOut, StayIn
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Observation {
    weather: Weather,
    wind: WindSpeed,
    pandemic: Pandemic,
    decision: Decision,
}

impl Observation {
    fn new(
    weather: Weather, wind: WindSpeed, pandemic: Pandemic, decision: Decision
) -> Observation {
        Observation {
            weather,
            wind,
            pandemic,
            decision
        }
    }
}

let observations = vec![
    Observation::new(Weather::Clear, 0.5, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Clear, 0.9, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Clear, 0.8, Pandemic::NoPandemic, Decision::StayIn),
    Observation::new(Weather::Stormy, 0.7, Pandemic::NoPandemic, Decision::StayIn),
    Observation::new(Weather::Rainy, 0.1, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Rainy, 0.5, Pandemic::NoPandemic, Decision::StayIn),
    Observation::new(Weather::Rainy, 0.6, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Rainy, 0.7, Pandemic::NoPandemic, Decision::StayIn),
    Observation::new(Weather::Cloudy, 0.3, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Cloudy, 0.5, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Cloudy, 0.2, Pandemic::NoPandemic, Decision::StayIn),
    Observation::new(Weather::Cloudy, 0.8, Pandemic::NoPandemic, Decision::StayIn),
    Observation::new(Weather::Sunny, 0.3, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Sunny, 0.9, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Sunny, 0.5, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Sunny, 0.5, Pandemic::NoPandemic, Decision::StayIn),
    Observation::new(Weather::Sunny, 0.5, Pandemic::Pandemic, Decision::StayIn),
    Observation::new(Weather::Clear, 0.1, Pandemic::Pandemic, Decision::StayIn),
    Observation::new(Weather::Clear, 0.9, Pandemic::Pandemic, Decision::StayIn)
];

fn predict(
    observations: &Vec<Observation>, weather: Weather, wind: WindSpeed, pandemic: Pandemic
) -> Decision {
    let total = observations.len() as f64;
    // first compute the number of each class in the data
    let total_stay_in = observations.iter()
        .filter(|observation| observation.decision == Decision::StayIn)
        .count() as f64;
    let total_go_out = observations.iter()
        .filter(|observation| observation.decision == Decision::GoOut)
        .count() as f64;

    let weather_log_probability_stay_in = {
        // compute how many rows in the data are this weather and stay in
        let total = observations.iter()
            .filter(|observation| observation.decision == Decision::StayIn)
            .filter(|observation| observation.weather == weather)
            .count() as f64;
        // there are 5 variants for the weather and we use laplacian smoothing
        // to avoid introducing zero probabilities, +1 / +5 treats the data
        // as if there is one stay in for each weather type.
        ((total + 1.0) / (total_stay_in + 5.0)).ln()
    };

    let weather_log_probability_go_out = {
        // compute how many rows in the data are this weather and go out
        let total = observations.iter()
            .filter(|observation| observation.decision == Decision::GoOut)
            .filter(|observation| observation.weather == weather)
            .count() as f64;
        // there are 5 variants for the weather and we use laplacian smoothing
        // to avoid introducing zero probabilities, +1 / +5 treats the data
        // as if there is one go out for each weather type.
        ((total + 1.0) / (total_go_out + 5.0)).ln()
    };

    // we're modelling the wind as a Gaussian so we get a probability by
    // computing the probability density for the wind we have, if it is
    // the same as the mean wind for stay in then it will be closest to 1
    // and the further away the closer to 0 it will be
    let wind_speed_model_stay_in: Gaussian<WindSpeed> = Gaussian::approximating(
        observations.iter()
        .filter(|observation| observation.decision == Decision::StayIn)
        .map(|observation| observation.wind)
    );
    let wind_log_probability_stay_in = wind_speed_model_stay_in.probability(&wind);

    let wind_speed_model_go_out: Gaussian<WindSpeed> = Gaussian::approximating(
        observations.iter()
        .filter(|observation| observation.decision == Decision::GoOut)
        .map(|observation| observation.wind)
    );
    let wind_log_probability_go_out = wind_speed_model_go_out.probability(&wind);

    let pandemic_log_probability_stay_in = {
        // compute how many rows in the data are this pandemic and stay in
        let total = observations.iter()
            .filter(|observation| observation.decision == Decision::StayIn)
            .filter(|observation| observation.pandemic == pandemic)
            .count() as f64;
        // there are 2 variants for the pandemic type and we use laplacian smoothing
        // to avoid introducing zero probabilities, +1 / +2 treats the data
        // as if there is one stay in for each pandemic type.
        ((total + 1.0) / (total_stay_in + 2.0)).ln()
    };

    let pandemic_log_probability_go_out = {
        // compute how many rows in the data are this pandemic and go out
        let total = observations.iter()
            .filter(|observation| observation.decision == Decision::GoOut)
            .filter(|observation| observation.pandemic == pandemic)
            .count() as f64;
        // there are 2 variants for the pandemic type and we use laplacian smoothing
        // to avoid introducing zero probabilities, +1 / +2 treats the data
        // as if there is one go out for each pandemic type.
        ((total + 1.0) / (total_go_out + 2.0)).ln()
    };

    let prior_log_probability_stay_in = total_stay_in / total;
    let prior_log_probability_go_out = total_go_out / total;

    let posterior_log_probability_stay_in = (
        prior_log_probability_stay_in
        + weather_log_probability_stay_in
        + wind_log_probability_stay_in
        + pandemic_log_probability_stay_in
    );
    let posterior_log_probability_go_out = (
        prior_log_probability_go_out
        + weather_log_probability_go_out
        + wind_log_probability_go_out
        + pandemic_log_probability_go_out
    );

    if posterior_log_probability_go_out > posterior_log_probability_stay_in {
        Decision::GoOut
    } else {
        Decision::StayIn
    }
}

let test_data = vec![
    Observation::new(Weather::Sunny, 0.8, Pandemic::NoPandemic, Decision::StayIn),
    Observation::new(Weather::Sunny, 0.2, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Stormy, 0.2, Pandemic::NoPandemic, Decision::StayIn),
    Observation::new(Weather::Cloudy, 0.3, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Rainy, 0.8, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Rainy, 0.5, Pandemic::NoPandemic, Decision::GoOut),
    Observation::new(Weather::Stormy, 0.6, Pandemic::Pandemic, Decision::StayIn),
    Observation::new(Weather::Rainy, 0.1, Pandemic::Pandemic, Decision::StayIn),
];

let predictions = test_data.iter()
    .map(|data| predict(&observations, data.weather, data.wind, data.pandemic))
    .collect::<Vec<Decision>>();

println!("Test data and predictions\n{:?}", test_data.iter()
    .cloned()
    .zip(predictions.clone())
    .collect::<Vec<(Observation, Decision)>>());

println!("Accuracy: {:?}", test_data.iter()
    .zip(predictions.clone())
    .map(|(data, decision)| if data.decision == decision { 1.0 } else { 0.0 })
    .sum::<f64>() / (test_data.len() as f64));

// To compute Recall and Precision it is neccessary to decide which class should be
// considered the positive and which should be the negative. For medical diagnosis
// this is always diagnosing something as present. For this example we take GoOut to
// be the positive case.

// True Positives are where the model predicts the positive class when it is
// the correct decision, eg in this scenario, going outside when it should
// decide to go outside
let true_positives = test_data.iter()
    .cloned()
    .zip(predictions.clone())
    .filter(|(data, _)| data.decision == Decision::GoOut)
    .filter(|(_, decision)| decision == &Decision::GoOut)
    .count() as f64;

// False Positives are when the model predicts the positive class when it is
// not the correct decision, eg in this scenario, going outside when it should
// decide to stay in
let false_positives = test_data.iter()
    .cloned()
    .zip(predictions.clone())
    .filter(|(data, _)| data.decision == Decision::StayIn)
    .filter(|(_, decision)| decision == &Decision::GoOut)
    .count() as f64;

// True Negatives are when the model predicts the negative class when it is
// the correct decision, eg in this scenario, staying in when it should
// decide to stay in
// there's a pandemic and no good reason to go outside
let true_negatives = test_data.iter()
    .cloned()
    .zip(predictions.clone())
    .filter(|(data, _)| data.decision == Decision::StayIn)
    .filter(|(_, decision)| decision == &Decision::StayIn)
    .count() as f64;

// False Negatives are when the model predicts the negative class when it is
// not the correct decision, eg in this scenario, staying in when it should
// decide to go outside
// there's a pandemic and no good reason to go outside
let false_negatives = test_data.iter()
    .cloned()
    .zip(predictions.clone())
    .filter(|(data, _)| data.decision == Decision::GoOut)
    .filter(|(_, decision)| decision == &Decision::StayIn)
    .count() as f64;

// Precision measures how good the model is at identifying the positive
// class (you can trivially get 100% precision by never predicting the
// positive class, as this means you can't get a false positive).
let precision = true_positives / (true_positives + false_positives);

// Recall is the true positive rate which is how good the model is at
// identifying the positive class out of all the positive cases (you can
// trivially get 100% recall by always predicting the positive class).
let recall = true_positives / (true_positives + false_negatives);

// The F-1 score is the harmonic mean of precision and recall and
// averages them for an accuracy measure.
// In this case the two classes are roughly equally likely, so the F-1
// score and Accuracy are similar. However, if the model had learned
// to always predict GoOut, then it would still have an accuracy of
// roughly 50% because of the equal likelihood, whereas its F-1 score
// would be much lower than 50%.
let f1_score = linear_algebra::f1_score(precision, recall);
println!("F1-Score: {:?}", f1_score);

assert!(f1_score > 0.8);
```

## 3 Class Naïve Bayes Example
[See submodule for 3 Class Naïve Bayes Example](three_class)
*/

#[rustfmt::skip]
pub mod three_class {
/*!
# 3 Class Naïve Bayes Example

For this example some population data is generated for a fictional alien race as I didn't
have any real datasets to hand. This alien race has 3 sexes (mysteriously no individuals
are ever intersex or trans) and is sexually dimorphic, meaning we can try to determine their
sex from various measuremnts.

As with humans, a gaussian distribution for physical charateristics is sensible due to
evolutionary and biological factors.

The example includes categorical features such as marking color and real valued
features such as height in order to show how both can be modelled with Naïve Bayes.

Note that most of the code below is for generating and clustering the data to perform
Naïve Bayes on, not doing it.

### Clustering

After generating the unlabelled data clustering is performed on a very small subset of that data
to find three cluster centres that are then used to assign the whole dataset sex class labels.
This creates a labelled dataset to perform Naïve Bayes on to see if we can predict sex given
the various features.

By clustering only a very small bit of the data, by chance we can expect there to be large gaps
in the subset because our data has many dimensions.

## Matrix APIs

```
use easy_ml::matrices::Matrix;
use easy_ml::matrices::slices::{Slice2D, Slice};
use easy_ml::linear_algebra;
use easy_ml::distributions::Gaussian;

use rand::{Rng, SeedableRng};
use rand::distributions::{DistIter, Standard};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, PartialEq, Debug)]
struct Alien {
    height: f64,
    primary_marking_color: AlienMarkingColor,
    tail_length: f64,
    metabolic_rate: f64,
    spiked_tail: bool,
    sex: AlienSex,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum AlienMarkingColor {
    Red = 1, Yellow, Orange, Blue, Purple, White, Black
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum AlienSex {
    A, B, C
}

/**
 * ===============================
 * =       Data Generation       =
 * ===============================
 */

// rather than explicitly define a generative function that already knows the relationship
// between alien charateristics instead the samples are generated without an assigned sex
// and then clustered using k-means

// use a fixed seed random generator from the rand crate
let mut random_generator = ChaCha8Rng::seed_from_u64(16);
let mut random_numbers: DistIter<Standard, &mut ChaCha8Rng, f64> =
    (&mut random_generator).sample_iter(Standard);

/**
 * Generates height data for creating the alien dataset.
 */
fn generate_heights(samples: usize, random_numbers: &mut impl Iterator<Item = f64>) -> Vec<f64> {
    // the average height shall be 1.5 meters with a standard deviation of 0.25
    let heights_distribution = Gaussian::new(1.5, 0.25 * 0.25);
    let mut heights = heights_distribution.draw(random_numbers, samples).unwrap();
    // ensure all aliens are at least 0.25 meters tall
    heights.drain(..).map(|x| x.max(0.25)).collect()
}

/**
 * Generates tail length data for creating the alien dataset.
 */
fn generate_tail_length(samples: usize, random_numbers: &mut impl Iterator<Item = f64>) -> Vec<f64> {
    // the average length shall be 1.25 meters with more variation in tail length
    let tails_distribution = Gaussian::new(1.25, 0.5 * 0.5);
    let mut tails = tails_distribution.draw(random_numbers, samples).unwrap();
    // ensure all tails are at least 0.5 meters long
    tails.drain(..).map(|x| if x > 0.5 { x } else { 0.5 }).collect()
}

/**
 * Generates color data for creating the alien dataset.
 *
 * Note that floats are still returned despite this being a category because we need all the
 * data types to be the same for clustering
 */
fn generate_colors(samples: usize, random_numbers: &mut impl Iterator<Item = f64>) -> Vec<f64> {
    let mut colors = Vec::with_capacity(samples);
    for i in 0..samples {
        let x = random_numbers.next().unwrap();
        if x < 0.2  {
            colors.push(AlienMarkingColor::Red as u8 as f64);
        } else if x < 0.3 {
            colors.push(AlienMarkingColor::Yellow as u8 as f64);
        } else if x < 0.45 {
            colors.push(AlienMarkingColor::Orange as u8 as f64);
        } else if x < 0.59 {
            colors.push(AlienMarkingColor::Blue as u8 as f64);
        } else if x < 0.63 {
            colors.push(AlienMarkingColor::Purple as u8 as f64);
        }  else if x < 0.9 {
            colors.push(AlienMarkingColor::White as u8 as f64);
        } else {
            colors.push(AlienMarkingColor::Black as u8 as f64);
        }
    }
    colors
}

/**
 * Recovers the color type which is the closest match to the input floating point color
 */
fn recover_generated_color(color: f64) -> AlienMarkingColor {
    let numerical_colors = [
        AlienMarkingColor::Red as u8 as f64,
        AlienMarkingColor::Yellow as u8 as f64,
        AlienMarkingColor::Orange as u8 as f64,
        AlienMarkingColor::Blue as u8 as f64,
        AlienMarkingColor::Purple as u8 as f64,
        AlienMarkingColor::White as u8 as f64,
        AlienMarkingColor::Black as u8 as f64,
    ];
    let colors = [
        AlienMarkingColor::Red,
        AlienMarkingColor::Yellow,
        AlienMarkingColor::Orange,
        AlienMarkingColor::Blue,
        AlienMarkingColor::Purple,
        AlienMarkingColor::White,
        AlienMarkingColor::Black,
    ];
    // look for the closest fit, as manipulated floating points may not be exact
    let color_index = numerical_colors.iter()
        // take the absolute difference so an exact match will become 0
        .map(|c| (c - color).abs())
        .enumerate()
        // find the element with the smallest difference in the list
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN should not be in list"))
        // discard the difference
        .map(|(index, difference)| index)
        // retrieve the value from the option
        .unwrap();
    colors[color_index]
}

/**
 * Generates metabolic rate data for creating the alien dataset.
 */
fn generate_metabolic_rate(
    samples: usize, random_numbers: &mut impl Iterator<Item = f64>
) -> Vec<f64> {
    // the average rate shall be 100 heart beats per minute with a standard deviation of 20
    let metabolic_rate_distribution = Gaussian::new(100.0, 20.0 * 20.0);
    let mut metabolic_rates = metabolic_rate_distribution.draw(random_numbers, samples).unwrap();
    // ensure all rates are at least 50 and less than 200
    metabolic_rates.drain(..).map(|x| x.max(50.0).min(200.0)).collect()
}

/**
 * Generates spiked tailness data for creating the alien dataset.
 */
fn generate_spiked_tail(
    samples: usize, random_numbers: &mut impl Iterator<Item = f64>
) -> Vec<f64> {
    let mut spikes = Vec::with_capacity(samples);
    for i in 0..samples {
        let x = random_numbers.next().unwrap();
        if x < 0.4 {
            // 0 shall represent a non spiked tail
            spikes.push(0.0)
        } else {
            // 1 shall represent a spiked tail
            spikes.push(1.0)
        }
    }
    spikes
}

/**
 * Recovers the spiked tail type which is the closest match to the input floating point
 */
fn recover_generated_spiked_tail(spiked: f64) -> bool {
    return spiked >= 0.5
}

// We shall generate 1000 samples for the dataset
const SAMPLES: usize = 1000;

// Collect all the float typed features into a matrix
let unlabelled_dataset = {
    let mut dataset = Matrix::column(generate_heights(SAMPLES, &mut random_numbers));
    dataset.insert_column_with(1,
        generate_colors(SAMPLES, &mut random_numbers).drain(..));
    dataset.insert_column_with(2,
        generate_tail_length(SAMPLES, &mut random_numbers).drain(..));
    dataset.insert_column_with(3,
        generate_metabolic_rate(SAMPLES, &mut random_numbers).drain(..));
    dataset.insert_column_with(4,
        generate_spiked_tail(SAMPLES, &mut random_numbers).drain(..));
    dataset
};

/**
 * ===============================
 * =          Clustering         =
 * ===============================
 */

// Create a subset of the first 30 samples from the full dataset to use for clustering
// TODO: Try to make this use MatrixRange here
let unlabelled_subset = unlabelled_dataset.retain(
    Slice2D::new()
    .columns(Slice::All())
    .rows(Slice::Range(0..30))
);

// We normalise all the features to 0 mean and 1 standard deviation because
// we will use euclidean distance as the distance matric, and our features
// have very different variances. This avoids the distance metric being
// dominated by any particular feature.

let mut means_and_variances = Vec::with_capacity(unlabelled_subset.columns());

// The normalised subset is computed taking the mean and variance from the subset,
// these means and variances will be needed later to apply to the rest of the data.
let mut normalised_subset = {
    let mut normalised_subset = unlabelled_subset;
    for feature in 0..normalised_subset.columns() {
        let mean = linear_algebra::mean(normalised_subset.column_iter(feature));
        let variance = linear_algebra::variance(normalised_subset.column_iter(feature));
        // save the data for normalisation and denormalisation for each feature
        // for use later
        means_and_variances.push(vec![ mean, variance ]);

        for row in 0..normalised_subset.rows() {
            let x = normalised_subset.get(row, feature);
            normalised_subset.set(row, feature, (x - mean) / variance);
        }
    }
    normalised_subset
};

// create a 5 x 2  matrix where each row is the mean and variance of each of the 5 features
let means_and_variances = Matrix::from(means_and_variances);

// pick the first 3 samples as the starting points for the cluster centres
// and place them into a 3 x 5 matrix where we have 3 rows of cluster centres
// and 5 features which are all normalised
let mut clusters = Matrix::from(vec![
    normalised_subset.row_iter(0).collect(),
    normalised_subset.row_iter(1).collect(),
    normalised_subset.row_iter(2).collect()]);

// add a 6th column to the subset to track the closest cluster for each sample
const CLUSTER_ID_COLUMN: usize = 5;
normalised_subset.insert_column(CLUSTER_ID_COLUMN, -1.0);

// set a threshold at which we consider the cluster centres to have converged
const CHANGE_THRESHOLD: f64 = 0.001;

// track how much the means have changed each update
let mut absolute_changes = -1.0;

// loop until we go under the CHANGE_THRESHOLD, reassigning points to the nearest
// cluster then cluster centres to their mean of points
while absolute_changes == -1.0 || absolute_changes > CHANGE_THRESHOLD {
    // assign each point to the nearest cluster centre by euclidean distance
    for point in 0..normalised_subset.rows() {
        let mut closest_cluster = -1.0;
        let mut least_squared_distance = std::f64::MAX;
        for cluster in 0..clusters.rows() {
            // we don't actually need to square root the distances for finding
            // which is least because least squared distance is the same as
            // least distance
            let squared_distance = {
                let mut sum = 0.0;
                for feature in 0..clusters.columns() {
                    let cluster_coordinate = clusters.get(cluster, feature);
                    let point_coordiante = normalised_subset.get(point, feature);
                    sum += (cluster_coordinate - point_coordiante).powi(2);
                }
                sum
            };

            if squared_distance < least_squared_distance {
                closest_cluster = cluster as f64;
                least_squared_distance = squared_distance;
            }
        }
        // save the cluster that is closest to each point in the 6th column
        normalised_subset.set(point, CLUSTER_ID_COLUMN, closest_cluster);
    }

    // update cluster centres to the mean of their points
    absolute_changes = 0.0;
    for cluster in 0..clusters.rows() {
        // construct a list of the points this cluster owns
        let owned = normalised_subset.column_iter(CLUSTER_ID_COLUMN)
            // zip together the id values with their index
            .enumerate()
            // exclude the points that aren't assigned to this cluster
            .filter(|(index, id)| (*id as usize) == cluster)
            // drop the cluster ids from each item and copy over the data
            // for each point for each feature
            .map(|(index, id)| {
                // for each point copy all its data in each feature excluding the
                // final cluster id column into a new vec
                normalised_subset.row_iter(index)
                    // taking the first 5 excludes the 6th column due to 0 indexing
                    .take(CLUSTER_ID_COLUMN)
                    .collect::<Vec<f64>>()
            })
            // collect into a vector of vectors containing each feature's data
            .collect::<Vec<Vec<f64>>>();
        // pass the vector of vectors into a matrix so we have
        // a matrix where each row is the data of a point this cluster owns
        let owned = Matrix::from(owned);

        // construct a vector of the mean for each feature that this cluster
        // now has
        let new_means = {
            let mut means = Vec::with_capacity(owned.rows());

            for feature in 0..owned.columns() {
                let mean = owned.column_iter(feature).sum::<f64>() / (owned.rows() as f64);
                means.push(mean);
            }

            means
        };

        // update each new mean for the cluster
        for feature in 0..clusters.columns() {
            let previous_mean = clusters.get(cluster, feature);
            // track the absolute difference between the new mean and the old one
            // so we know when to stop updating the clusters
            absolute_changes += (previous_mean - new_means[feature]).abs();

            println!(
                "Cluster {:?} mean for feature {:?} is now {:?} was {:?}",
                cluster, feature, new_means[feature], previous_mean
            );
            clusters.set(cluster, feature, new_means[feature]);
        }
    }
}

println!(
    "Denormalised clusters at convergence:\n{:?}\n{:.3}",
    vec![ "H", "C", "T", "M", "S" ],
    clusters.map_with_index(|x, _cluster, feature| {
        let mean = means_and_variances.get(feature, 0);
        let variance = means_and_variances.get(feature, 1);
        (x * variance) + mean
    }));

// Now we will assign every alien in the full dataset a sex using these cluster centres
let mut aliens: Vec<Alien> = Vec::with_capacity(unlabelled_dataset.rows());

fn assign_alien_sex(index: u8) -> AlienSex {
    if index == 0 {
        AlienSex::A
    } else if index == 1 {
        AlienSex::B
    } else {
        AlienSex::C
    }
}

for i in 0..unlabelled_dataset.rows() {
    let alien_data = Matrix::column(unlabelled_dataset.row_iter(i).collect());
    // normalise the alien data first so comparisons are on unit variance
    // and zero mean
    let normalised_alien_data = alien_data.map_with_index(|x, feature, _| {
        let mean = means_and_variances.get(feature, 0);
        let variance = means_and_variances.get(feature, 1);
        // normalise each feature in the alien data
        (x - mean) / variance
    });
    let mut distances = Vec::with_capacity(clusters.rows());
    for j in 0..clusters.rows() {
        let cluster_data = Matrix::row(clusters.row_iter(j).collect());
        // use euclidean distance to compare the alien with the cluster, the cluster
        // is already normalised
        let sum_of_squares = (cluster_data * &normalised_alien_data).scalar();
        distances.push(sum_of_squares);
    }

    // find the cluster with the lowest distance to each point and get its index
    let chosen_cluster = distances.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN should not be in list"))
        .map(|(i, _)| i)
        .unwrap();

    // convert each float to the correct data type
    aliens.push(Alien {
        height: alien_data.get(0, 0),
        primary_marking_color: recover_generated_color(alien_data.get(1, 0)),
        tail_length: alien_data.get(2, 0),
        metabolic_rate: alien_data.get(3, 0),
        spiked_tail: recover_generated_spiked_tail(alien_data.get(4, 0)),
        sex: assign_alien_sex(chosen_cluster as u8),
    })
}

println!("First 10 aliens");
for i in 0..10 {
    println!("{:?}", aliens[i]);
}

// Put the aliens in a matrix for convenience
let aliens = Matrix::row(aliens);

println!("Sex A aliens total: {}", aliens.row_reference_iter(0)
    .fold(0, |accumulator, alien| accumulator + if alien.sex == AlienSex::A { 1 } else { 0 }));

println!("Sex B aliens total: {}", aliens.row_reference_iter(0)
    .fold(0, |accumulator, alien| accumulator + if alien.sex == AlienSex::B { 1 } else { 0 }));

println!("Sex C aliens total: {}", aliens.row_reference_iter(0)
    .fold(0, |accumulator, alien| accumulator + if alien.sex == AlienSex::C { 1 } else { 0 }));

/**
 * ===============================
 * =         Naïve Bayes         =
 * ===============================
 */

// Each class is roughly one third so we should not have a strong prior to a particular class

// In order to evaluate the performance of the Naïve Bayes classifier we will hold out
// the last 100 aliens from the dataset and use them as training data

let training_data = Matrix::column(aliens.row_iter(0).take(900).collect());
let test_data = Matrix::column(aliens.row_iter(0).skip(900).collect());

/**
 * Predicts the most probable alien sex for each test input alien (disregarding
 * the sex field in those inputs)
 *
 * For the real valued features the probabilities are computed by modelling
 * the features (conditioned on each class) as gaussian distributions.
 * For categorical features laplacian smoothing of the counts is used to
 * estimate probabilities of the features (conditioned on each class).
 */
fn predict_aliens(training_data: &Matrix<Alien>, test_data: &Matrix<Alien>) -> Matrix<AlienSex> {
    let mut relative_log_probabilities = Vec::with_capacity(3);

    for class in &[ AlienSex::A, AlienSex::B, AlienSex::C ] {
        let training_data_class_only = training_data.column_iter(0)
            .filter(|a| &a.sex == class)
            .collect::<Vec<Alien>>();

        // compute how likely each class is in the training set
        let prior = (training_data_class_only.len() as f64) / (training_data.rows() as f64);

        // We model the real valued features as Gaussians, note that these
        // are Gaussian distributions over only the training data of each class
        let heights: Gaussian<f64> = Gaussian::approximating(
            training_data_class_only.iter().map(|a| a.height)
        );
        let tail_lengths: Gaussian<f64> = Gaussian::approximating(
            training_data_class_only.iter().map(|a| a.tail_length)
        );
        let metabolic_rates: Gaussian<f64> = Gaussian::approximating(
            training_data_class_only.iter().map(|a| a.metabolic_rate)
        );

        // gradually build up the sum of log probabilities to get the
        // log of the prior * likelihood which will be proportional to the posterior
        let relative_log_probabilities_of_class = test_data.column_reference_iter(0)
        .map(|alien| {
            // probabilitiy of the alien sex and the alien
            let mut log_relative_probability = prior.ln();

            // Compute the probability using the Gaussian model for each real valued feature.
            // Due to floating point precision limits and the variance for some of these
            // Gaussian models being extremely small (0.01 for heights) we
            // check if a probability computed is zero or extremely close to zero
            // and if so increase it a bit to avoid computing -inf when we take the log.

            let mut height_given_class = heights.probability(&alien.height);
            if height_given_class.abs() <= 0.000000000001 {
                height_given_class = 0.000000000001;
            }
            log_relative_probability += height_given_class.ln();

            let mut tail_given_class = tail_lengths.probability(&alien.tail_length);
            if tail_given_class.abs() <= 0.000000000001 {
                tail_given_class = 0.000000000001;
            }
            log_relative_probability += tail_given_class.ln();

            let mut metabolic_rates_given_class = metabolic_rates.probability(
                &alien.metabolic_rate);
            if metabolic_rates_given_class.abs() <= 0.000000000001 {
                metabolic_rates_given_class = 0.000000000001;
            }
            log_relative_probability += metabolic_rates_given_class.ln();

            // compute the probability of the categorical features using lapacian smoothing
            let color_of_class = training_data_class_only.iter()
                .map(|a| a.primary_marking_color)
                // count how many aliens of this class have this color
                .fold(0, |acc, color|
                    acc + if color == alien.primary_marking_color { 1 } else { 0 });
            // with laplacian smoothing we assume there is one datapoint for each color
            // which avoids zero probabilities but does not distort the probabilities much
            // there are 7 color types so we add 7 to the total
            let color_given_class = ((color_of_class + 1) as f64)
                / ((training_data_class_only.len() + 7) as f64);
            log_relative_probability += color_given_class.ln();

            let spiked_tail_of_class = training_data_class_only.iter()
                .map(|a| a.spiked_tail)
                // count how many aliens of this class have a spiked tail or not
                .fold(0, |acc, spiked| acc + if spiked == alien.spiked_tail { 1 } else { 0 });
            // again we assume one alien of the class with a spiked tail and one without
            // to avoid zero probabilities
            let spiked_tail_given_class = ((spiked_tail_of_class + 1) as f64)
                / ((training_data_class_only.len() + 2) as f64);
            log_relative_probability += spiked_tail_given_class.ln();

            if log_relative_probability == std::f64::NEG_INFINITY {
                println!("Individual probs P:{} H:{} T:{} M:{} C:{} S:{}",
                    prior, height_given_class, tail_given_class, metabolic_rates_given_class,
                    color_given_class, spiked_tail_given_class);
            }

            log_relative_probability
        }).collect();

        relative_log_probabilities.push(relative_log_probabilities_of_class);
    }

    // collect the relative probabilitiy estimates for each class and each alien
    // into a 3 x 100 matrix respectively
    let probabilities = Matrix::from(relative_log_probabilities);

    let predictions = (0..probabilities.columns()).map(|i| {
        let predicted_class_index = probabilities.column_iter(i)
            .enumerate()
            // find the class with the highest relative probability estimate
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN should not be in list"))
            // discard the probability
            .map(|(index, p)| index)
            // retrieve the value from the option
            .unwrap();
        if predicted_class_index == 0 {
            AlienSex::A
        } else if predicted_class_index == 1 {
            AlienSex::B
        } else {
            AlienSex::C
        }
    }).collect();

    Matrix::column(predictions)
}

let predictions = predict_aliens(&training_data, &test_data);

println!("First 10 test aliens and predictions");
for i in 0..10 {
    println!("Predicted: {:?} Input: {:?}", predictions.get(i, 0), test_data.get(i, 0));
}

let accuracy = test_data.column_iter(0)
    .zip(predictions.column_iter(0))
    .map(|(alien, prediction)| if alien.sex == prediction { 1 } else { 0 })
    .sum::<u16>() as f64 / (test_data.rows() as f64);

println!("Accuracy {}%", accuracy * 100.0);

/**
 * ===============================
 * =           Analysis          =
 * ===============================
 */

// We can get a better sense of how well our classifier has done by
// printing the confusion matrix

// Construct a confusion matrix of actual x predicted classes, using A as 0, B as 1 and C as 2
// for indexing. If the accuracy was 100% we would see only non zero numbers on the diagonal
// as every prediction would be the actual class.
let confusion_matrix = {
    let mut confusion_matrix = Matrix::empty(0, (3, 3));

    // loop through all the actual and predicted classes to fill the confusion matrix
    // with the total occurances of each possible combination
    for (actual, predicted) in test_data.column_iter(0).zip(predictions.column_iter(0)) {
        match actual.sex {
            AlienSex::A => {
                match predicted {
                    AlienSex::A => confusion_matrix.set(0, 0, confusion_matrix.get(0, 0) + 1),
                    AlienSex::B => confusion_matrix.set(0, 1, confusion_matrix.get(0, 1) + 1),
                    AlienSex::C => confusion_matrix.set(0, 2, confusion_matrix.get(0, 2) + 1),
                }
            },
            AlienSex::B => {
                match predicted {
                    AlienSex::A => confusion_matrix.set(1, 0, confusion_matrix.get(1, 0) + 1),
                    AlienSex::B => confusion_matrix.set(1, 1, confusion_matrix.get(1, 1) + 1),
                    AlienSex::C => confusion_matrix.set(1, 2, confusion_matrix.get(1, 2) + 1),
                }
            },
            AlienSex::C => {
                match predicted {
                    AlienSex::A => confusion_matrix.set(2, 0, confusion_matrix.get(2, 0) + 1),
                    AlienSex::B => confusion_matrix.set(2, 1, confusion_matrix.get(2, 1) + 1),
                    AlienSex::C => confusion_matrix.set(2, 2, confusion_matrix.get(2, 2) + 1),
                }
            }
        }
    }

    confusion_matrix
};

println!("Confusion matrix: Rows are actual class, Columns are predicted class\n{}",
    confusion_matrix);
println!("  A  B  C");
```

## Tensor APIs


```
use easy_ml::tensors::Tensor;
use easy_ml::tensors::views::{TensorView, TensorStack};
use easy_ml::linear_algebra;
use easy_ml::distributions::Gaussian;

use rand::{Rng, SeedableRng};
use rand::distributions::{DistIter, Standard};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, PartialEq, Debug)]
struct Alien {
    height: f64,
    primary_marking_color: AlienMarkingColor,
    tail_length: f64,
    metabolic_rate: f64,
    spiked_tail: bool,
    sex: AlienSex,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum AlienMarkingColor {
    Red = 1, Yellow, Orange, Blue, Purple, White, Black
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum AlienSex {
    A, B, C
}

/**
 * ===============================
 * =       Data Generation       =
 * ===============================
 */

// rather than explicitly define a generative function that already knows the relationship
// between alien charateristics instead the samples are generated without an assigned sex
// and then clustered using k-means

// use a fixed seed random generator from the rand crate
let mut random_generator = ChaCha8Rng::seed_from_u64(16);
let mut random_numbers: DistIter<Standard, &mut ChaCha8Rng, f64> =
    (&mut random_generator).sample_iter(Standard);

/**
 * Generates height data for creating the alien dataset.
 */
fn generate_heights(samples: usize, random_numbers: &mut impl Iterator<Item = f64>) -> Vec<f64> {
    // the average height shall be 1.5 meters with a standard deviation of 0.25
    let heights_distribution = Gaussian::new(1.5, 0.25 * 0.25);
    let mut heights = heights_distribution.draw(random_numbers, samples).unwrap();
    // ensure all aliens are at least 0.25 meters tall
    heights.drain(..).map(|x| x.max(0.25)).collect()
}

/**
 * Generates tail length data for creating the alien dataset.
 */
fn generate_tail_length(samples: usize, random_numbers: &mut impl Iterator<Item = f64>) -> Vec<f64> {
    // the average length shall be 1.25 meters with more variation in tail length
    let tails_distribution = Gaussian::new(1.25, 0.5 * 0.5);
    let mut tails = tails_distribution.draw(random_numbers, samples).unwrap();
    // ensure all tails are at least 0.5 meters long
    tails.drain(..).map(|x| if x > 0.5 { x } else { 0.5 }).collect()
}

/**
 * Generates color data for creating the alien dataset.
 *
 * Note that floats are still returned despite this being a category because we need all the
 * data types to be the same for clustering
 */
fn generate_colors(samples: usize, random_numbers: &mut impl Iterator<Item = f64>) -> Vec<f64> {
    let mut colors = Vec::with_capacity(samples);
    for i in 0..samples {
        let x = random_numbers.next().unwrap();
        if x < 0.2  {
            colors.push(AlienMarkingColor::Red as u8 as f64);
        } else if x < 0.3 {
            colors.push(AlienMarkingColor::Yellow as u8 as f64);
        } else if x < 0.45 {
            colors.push(AlienMarkingColor::Orange as u8 as f64);
        } else if x < 0.59 {
            colors.push(AlienMarkingColor::Blue as u8 as f64);
        } else if x < 0.63 {
            colors.push(AlienMarkingColor::Purple as u8 as f64);
        }  else if x < 0.9 {
            colors.push(AlienMarkingColor::White as u8 as f64);
        } else {
            colors.push(AlienMarkingColor::Black as u8 as f64);
        }
    }
    colors
}

/**
 * Recovers the color type which is the closest match to the input floating point color
 */
fn recover_generated_color(color: f64) -> AlienMarkingColor {
    let numerical_colors = [
        AlienMarkingColor::Red as u8 as f64,
        AlienMarkingColor::Yellow as u8 as f64,
        AlienMarkingColor::Orange as u8 as f64,
        AlienMarkingColor::Blue as u8 as f64,
        AlienMarkingColor::Purple as u8 as f64,
        AlienMarkingColor::White as u8 as f64,
        AlienMarkingColor::Black as u8 as f64,
    ];
    let colors = [
        AlienMarkingColor::Red,
        AlienMarkingColor::Yellow,
        AlienMarkingColor::Orange,
        AlienMarkingColor::Blue,
        AlienMarkingColor::Purple,
        AlienMarkingColor::White,
        AlienMarkingColor::Black,
    ];
    // look for the closest fit, as manipulated floating points may not be exact
    let color_index = numerical_colors.iter()
        // take the absolute difference so an exact match will become 0
        .map(|c| (c - color).abs())
        .enumerate()
        // find the element with the smallest difference in the list
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN should not be in list"))
        // discard the difference
        .map(|(index, difference)| index)
        // retrieve the value from the option
        .unwrap();
    colors[color_index]
}

/**
 * Generates metabolic rate data for creating the alien dataset.
 */
fn generate_metabolic_rate(
    samples: usize, random_numbers: &mut impl Iterator<Item = f64>
) -> Vec<f64> {
    // the average rate shall be 100 heart beats per minute with a standard deviation of 20
    let metabolic_rate_distribution = Gaussian::new(100.0, 20.0 * 20.0);
    let mut metabolic_rates = metabolic_rate_distribution.draw(random_numbers, samples).unwrap();
    // ensure all rates are at least 50 and less than 200
    metabolic_rates.drain(..).map(|x| x.max(50.0).min(200.0)).collect()
}

/**
 * Generates spiked tailness data for creating the alien dataset.
 */
fn generate_spiked_tail(
    samples: usize, random_numbers: &mut impl Iterator<Item = f64>
) -> Vec<f64> {
    let mut spikes = Vec::with_capacity(samples);
    for i in 0..samples {
        let x = random_numbers.next().unwrap();
        if x < 0.4 {
            // 0 shall represent a non spiked tail
            spikes.push(0.0)
        } else {
            // 1 shall represent a spiked tail
            spikes.push(1.0)
        }
    }
    spikes
}

/**
 * Recovers the spiked tail type which is the closest match to the input floating point
 */
fn recover_generated_spiked_tail(spiked: f64) -> bool {
    return spiked >= 0.5
}

// We shall generate 1000 samples for the dataset
const SAMPLES: usize = 1000;

// Each sample has 5 features
const FEATURES: usize = 5;

// Collect all the float typed features into a 2 dimensional tensor of samples x data
let unlabelled_dataset = {
    let heights = Tensor::from(
        [("sample", SAMPLES)], generate_heights(SAMPLES, &mut random_numbers)
    );
    let colours = Tensor::from(
        [("sample", SAMPLES)], generate_colors(SAMPLES, &mut random_numbers)
    );
    let tail_lengths = Tensor::from(
        [("sample", SAMPLES)], generate_tail_length(SAMPLES, &mut random_numbers)
    );
    let metabolic_rates = Tensor::from(
        [("sample", SAMPLES)], generate_metabolic_rate(SAMPLES, &mut random_numbers)
    );
    let spiked_tail = Tensor::from(
        [("sample", SAMPLES)], generate_spiked_tail(SAMPLES, &mut random_numbers)
    );

    TensorView::from(
        TensorStack::<_, [_; 5], 1>::from(
            [ heights, colours, tail_lengths, metabolic_rates, spiked_tail ],
            // we stack along the second axis that we're creating to get our rows of samples
            // this leaves the existing axis in all our individual column vectors to be each
            // feature's data
            (1, "feature")
        )
    )
};

/**
 * ===============================
 * =          Clustering         =
 * ===============================
 */

// Create a subset of the first 30 samples from the full dataset to use for clustering
const SUBSET_SAMPLES: usize = 30;
let unlabelled_subset = unlabelled_dataset.range([("sample", 0..SUBSET_SAMPLES)])
    .expect("The unlabelled data set should have at least 30 samples");

// We normalise all the features to 0 mean and 1 standard deviation because
// we will use euclidean distance as the distance matric, and our features
// have very different variances. This avoids the distance metric being
// dominated by any particular feature.

let mut means_and_variances = Vec::with_capacity(FEATURES);

// The normalised subset is computed taking the mean and variance from the subset,
// these means and variances will be needed later to apply to the rest of the data.
let mut normalised_subset = {
    let mut normalised_subset = unlabelled_subset.map(|x| x);
    for feature in 0..FEATURES {
        let mut feature_data = normalised_subset.select_mut([("feature", feature)]);
        let mean = linear_algebra::mean(feature_data.iter());
        let variance = linear_algebra::variance(feature_data.iter());
        // save the data for normalisation and denormalisation for each feature
        // for use later
        means_and_variances.push((mean, variance));

        for (row, x) in feature_data.iter_reference_mut().enumerate() {
            *x = (*x - mean) / variance;
        }
    }
    normalised_subset
};

// create a tensor where each row is a tuple of the mean and variance of each of the
// 5 features
let means_and_variances = Tensor::from([("feature", FEATURES)], means_and_variances);

// pick the first 3 samples as the starting points for the cluster centres
// and place them into a 3 x 5 tensor where we have 3 rows of cluster centres
// and 5 features which are all normalised
const CLUSTERS: usize = 3;
let mut clusters = TensorView::from(
    TensorStack::<_, [_; 3], 1>::from(
        [
            normalised_subset.select([("sample", 0)]).source(),
            normalised_subset.select([("sample", 1)]).source(),
            normalised_subset.select([("sample", 2)]).source()
        ],
        (0, "cluster")
    )
).map(|x| x);

// add a 6th column to the subset to track the closest cluster for each sample
const CLUSTER_ID_COLUMN: usize = 5;
let mut normalised_subset = {
    let mut data = normalised_subset.iter();
    let mut tensor = Tensor::empty([("sample", SUBSET_SAMPLES), ("feature", FEATURES + 1)], -1.0);
    for ([_row, feature], x) in tensor.iter_reference_mut().with_index() {
        *x = match feature {
            0 | 1 | 2 | 3 | 4 => data.next().unwrap(),
            _ => -1.0
        };
    }
    tensor
};

// set a threshold at which we consider the cluster centres to have converged
const CHANGE_THRESHOLD: f64 = 0.001;

// track how much the means have changed each update
let mut absolute_changes = -1.0;

// loop until we go under the CHANGE_THRESHOLD, reassigning points to the nearest
// cluster then cluster centres to their mean of points
while absolute_changes == -1.0 || absolute_changes > CHANGE_THRESHOLD {
    // assign each point to the nearest cluster centre by euclidean distance
    for point in 0..SUBSET_SAMPLES {
        let mut closest_cluster = -1.0;
        let mut least_squared_distance = std::f64::MAX;
        let clusters_indexing = clusters.index();
        let mut samples_indexing = normalised_subset.index_mut();
        for cluster in 0..CLUSTERS {
            // we don't actually need to square root the distances for finding
            // which is least because least squared distance is the same as
            // least distance
            let squared_distance = {
                let mut sum = 0.0;

                for feature in 0..FEATURES {
                    let cluster_coordinate = clusters_indexing.get([cluster, feature]);
                    let point_coordiante = samples_indexing.get([point, feature]);
                    sum += (cluster_coordinate - point_coordiante).powi(2);
                }
                sum
            };

            if squared_distance < least_squared_distance {
                closest_cluster = cluster as f64;
                least_squared_distance = squared_distance;
            }
        }
        // save the cluster that is closest to each point in the 6th column
        *samples_indexing.get_ref_mut([point, CLUSTER_ID_COLUMN]) = closest_cluster;
    }

    // update cluster centres to the mean of their points
    absolute_changes = 0.0;
    for cluster in 0..CLUSTERS {
        // construct a list of the points this cluster owns
        let owned = normalised_subset.select([("feature", CLUSTER_ID_COLUMN)]).iter()
            // zip together the id values with their index
            .enumerate()
            // exclude the points that aren't assigned to this cluster
            .filter(|(index, id)| (*id as usize) == cluster)
            // drop the cluster ids from each item and copy over the data
            // for each point for each feature
            .map(|(index, id)| {
                // for each point copy all its data in each feature excluding the
                // final cluster id column into a new vec
                normalised_subset.select([("sample", index)]).iter()
                    // taking the first 5 excludes the 6th column due to 0 indexing
                    .take(CLUSTER_ID_COLUMN)
                    .collect::<Vec<f64>>()
            })
            // collect into a vector of vectors containing each feature's data
            .collect::<Vec<Vec<f64>>>();
        // pass the vector of vectors into a tensor so we have
        // a tensor where each row is the data of a point this cluster owns
        let owned_points = owned.len();
        let owned = Tensor::from([("owned", owned.len())], owned);

        // construct a vector of the mean for each feature that this cluster
        // now has
        let new_means = {
            let mut means = Vec::with_capacity(owned_points);

            for feature in 0..FEATURES {
                let mean = owned.iter().map(|x| x[feature]).sum::<f64>() / (owned_points as f64);
                means.push(mean);
            }

            means
        };

        // update each new mean for the cluster
        for feature in 0..FEATURES {
            let mut clusters_indexing = clusters.index_mut();
            let previous_mean = clusters_indexing.get([cluster, feature]);
            // track the absolute difference between the new mean and the old one
            // so we know when to stop updating the clusters
            absolute_changes += (previous_mean - new_means[feature]).abs();

            println!(
                "Cluster {:?} mean for feature {:?} is now {:?} was {:?}",
                cluster, feature, new_means[feature], previous_mean
            );
            *clusters_indexing.get_ref_mut([cluster, feature]) = new_means[feature];
        }
    }
}

println!(
    "Denormalised clusters at convergence:\n{:?}\n{:.3}",
    vec![ "H", "C", "T", "M", "S" ],
    clusters.map_with_index(|[_cluster, feature], x| {
        let indexing = means_and_variances.index();
        let (mean, variance) = indexing.get([feature]);
        (x * variance) + mean
    }));

// Now we will assign every alien in the full dataset a sex using these cluster centres
let mut aliens: Vec<Alien> = Vec::with_capacity(SAMPLES);

fn assign_alien_sex(index: u8) -> AlienSex {
    if index == 0 {
        AlienSex::A
    } else if index == 1 {
        AlienSex::B
    } else {
        AlienSex::C
    }
}

for i in 0..SAMPLES {
    let alien_data = unlabelled_dataset.select([("sample", i)]).map(|x| x);
    // normalise the alien data first so comparisons are on unit variance
    // and zero mean
    let normalised_alien_data = alien_data.map_with_index(|[feature], x| {
        let indexing = means_and_variances.index();
        let (mean, variance) = indexing.get([feature]);
        // normalise each feature in the alien data
        (x - mean) / variance
    });
    let mut distances = Vec::with_capacity(CLUSTERS);
    for j in 0..CLUSTERS {
        let cluster_data = clusters.select([("cluster", j)]);
        // use euclidean distance to compare the alien with the cluster, the cluster
        // is already normalised
        // we have a 1 x 5 tensor for cluser data and a 5 x 1 tensor for alien data so matrix
        // multiplication here gives us the sum of products
        let sum_of_squares = (
            cluster_data.expand([(0, "samples")]) * normalised_alien_data.expand([(1, "alien")])
        ).first();
        distances.push(sum_of_squares);
    }

    // find the cluster with the lowest distance to each point and get its index
    let chosen_cluster = distances.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN should not be in list"))
        .map(|(i, _)| i)
        .unwrap();

    // convert each float to the correct data type
    let alien_data_indexing = alien_data.index();
    aliens.push(Alien {
        height: alien_data_indexing.get([0]),
        primary_marking_color: recover_generated_color(alien_data_indexing.get([1])),
        tail_length: alien_data_indexing.get([2]),
        metabolic_rate: alien_data_indexing.get([3]),
        spiked_tail: recover_generated_spiked_tail(alien_data_indexing.get([4])),
        sex: assign_alien_sex(chosen_cluster as u8),
    })
}

println!("First 10 aliens");
for i in 0..10 {
    println!("{:?}", aliens[i]);
}

// Put the aliens in a tensor for convenience
let aliens = Tensor::from([("sample", SAMPLES)], aliens);

println!("Sex A aliens total: {}", aliens.iter_reference()
    .fold(0, |accumulator, alien| accumulator + if alien.sex == AlienSex::A { 1 } else { 0 }));

println!("Sex B aliens total: {}", aliens.iter_reference()
    .fold(0, |accumulator, alien| accumulator + if alien.sex == AlienSex::B { 1 } else { 0 }));

println!("Sex C aliens total: {}", aliens.iter_reference()
    .fold(0, |accumulator, alien| accumulator + if alien.sex == AlienSex::C { 1 } else { 0 }));

/**
 * ===============================
 * =         Naïve Bayes         =
 * ===============================
 */

// Each class is roughly one third so we should not have a strong prior to a particular class

// In order to evaluate the performance of the Naïve Bayes classifier we will hold out
// the last 100 aliens from the dataset and use them as training data

const TRAINING_SAMPLES: usize = 900;
const TESTING_SAMPLES: usize = 100;
let training_data = Tensor::from([("sample", 900)], aliens.iter().take(TRAINING_SAMPLES).collect());
let test_data = Tensor::from([("sample", 100)], aliens.iter().skip(TRAINING_SAMPLES).collect());

/**
 * Predicts the most probable alien sex for each test input alien (disregarding
 * the sex field in those inputs)
 *
 * For the real valued features the probabilities are computed by modelling
 * the features (conditioned on each class) as gaussian distributions.
 * For categorical features laplacian smoothing of the counts is used to
 * estimate probabilities of the features (conditioned on each class).
 */
fn predict_aliens(training_data: &Tensor<Alien, 1>, test_data: &Tensor<Alien, 1>) -> Tensor<AlienSex, 1> {
    let mut relative_log_probabilities: Vec<f64> = Vec::with_capacity(300);

    for class in &[ AlienSex::A, AlienSex::B, AlienSex::C ] {
        let training_data_class_only = training_data.iter()
            .filter(|a| &a.sex == class)
            .collect::<Vec<Alien>>();

        // compute how likely each class is in the training set
        let prior = (training_data_class_only.len() as f64) / (TRAINING_SAMPLES as f64);

        // We model the real valued features as Gaussians, note that these
        // are Gaussian distributions over only the training data of each class
        let heights: Gaussian<f64> = Gaussian::approximating(
            training_data_class_only.iter().map(|a| a.height)
        );
        let tail_lengths: Gaussian<f64> = Gaussian::approximating(
            training_data_class_only.iter().map(|a| a.tail_length)
        );
        let metabolic_rates: Gaussian<f64> = Gaussian::approximating(
            training_data_class_only.iter().map(|a| a.metabolic_rate)
        );

        // gradually build up the sum of log probabilities to get the
        // log of the prior * likelihood which will be proportional to the posterior
        let mut relative_log_probabilities_of_class = test_data.iter_reference()
        .map(|alien| {
            // probabilitiy of the alien sex and the alien
            let mut log_relative_probability = prior.ln();

            // Compute the probability using the Gaussian model for each real valued feature.
            // Due to floating point precision limits and the variance for some of these
            // Gaussian models being extremely small (0.01 for heights) we
            // check if a probability computed is zero or extremely close to zero
            // and if so increase it a bit to avoid computing -inf when we take the log.

            let mut height_given_class = heights.probability(&alien.height);
            if height_given_class.abs() <= 0.000000000001 {
                height_given_class = 0.000000000001;
            }
            log_relative_probability += height_given_class.ln();

            let mut tail_given_class = tail_lengths.probability(&alien.tail_length);
            if tail_given_class.abs() <= 0.000000000001 {
                tail_given_class = 0.000000000001;
            }
            log_relative_probability += tail_given_class.ln();

            let mut metabolic_rates_given_class = metabolic_rates.probability(
                &alien.metabolic_rate);
            if metabolic_rates_given_class.abs() <= 0.000000000001 {
                metabolic_rates_given_class = 0.000000000001;
            }
            log_relative_probability += metabolic_rates_given_class.ln();

            // compute the probability of the categorical features using lapacian smoothing
            let color_of_class = training_data_class_only.iter()
                .map(|a| a.primary_marking_color)
                // count how many aliens of this class have this color
                .fold(0, |acc, color|
                    acc + if color == alien.primary_marking_color { 1 } else { 0 });
            // with laplacian smoothing we assume there is one datapoint for each color
            // which avoids zero probabilities but does not distort the probabilities much
            // there are 7 color types so we add 7 to the total
            let color_given_class = ((color_of_class + 1) as f64)
                / ((training_data_class_only.len() + 7) as f64);
            log_relative_probability += color_given_class.ln();

            let spiked_tail_of_class = training_data_class_only.iter()
                .map(|a| a.spiked_tail)
                // count how many aliens of this class have a spiked tail or not
                .fold(0, |acc, spiked| acc + if spiked == alien.spiked_tail { 1 } else { 0 });
            // again we assume one alien of the class with a spiked tail and one without
            // to avoid zero probabilities
            let spiked_tail_given_class = ((spiked_tail_of_class + 1) as f64)
                / ((training_data_class_only.len() + 2) as f64);
            log_relative_probability += spiked_tail_given_class.ln();

            if log_relative_probability == std::f64::NEG_INFINITY {
                println!("Individual probs P:{} H:{} T:{} M:{} C:{} S:{}",
                    prior, height_given_class, tail_given_class, metabolic_rates_given_class,
                    color_given_class, spiked_tail_given_class);
            }

            log_relative_probability
        }).collect();

        relative_log_probabilities.append(&mut relative_log_probabilities_of_class);
    }

    // collect the relative probabilitiy estimates for each class and each alien
    // into a 3 x 100 matrix respectively
    let probabilities = Tensor::from([("class", 3), ("sample", 100)], relative_log_probabilities);

    let predictions = (0..100).map(|i| {
        let predicted_class_index = probabilities.select([("sample", i)])
            .iter()
            .enumerate()
            // find the class with the highest relative probability estimate
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN should not be in list"))
            // discard the probability
            .map(|(index, p)| index)
            // retrieve the value from the option
            .unwrap();
        if predicted_class_index == 0 {
            AlienSex::A
        } else if predicted_class_index == 1 {
            AlienSex::B
        } else {
            AlienSex::C
        }
    }).collect();

    Tensor::from([("class", 100)], predictions)
}

let predictions = predict_aliens(&training_data, &test_data);

println!("First 10 test aliens and predictions");
for i in 0..10 {
    println!("Predicted: {:?} Input: {:?}", predictions.index().get([i]), test_data.index().get([i]));
}

let accuracy = test_data.iter_reference()
    .zip(predictions.iter_reference())
    .map(|(alien, prediction)| if alien.sex == *prediction { 1 } else { 0 })
    .sum::<u16>() as f64 / (TESTING_SAMPLES as f64);

println!("Accuracy {}%", accuracy * 100.0);

/**
 * ===============================
 * =           Analysis          =
 * ===============================
 */

// We can get a better sense of how well our classifier has done by
// printing the confusion matrix

// Construct a confusion matrix of actual x predicted classes, using A as 0, B as 1 and C as 2
// for indexing. If the accuracy was 100% we would see only non zero numbers on the diagonal
// as every prediction would be the actual class.
let confusion_matrix = {
    let mut confusion_matrix = Tensor::empty([("actual_class", 3), ("predicted_class", 3)], 0);
    let mut confusion_matrix_indexing = confusion_matrix.index_mut();

    // loop through all the actual and predicted classes to fill the confusion matrix
    // with the total occurances of each possible combination
    for (actual, predicted) in test_data.iter_reference().zip(predictions.iter_reference()) {
        match actual.sex {
            AlienSex::A => {
                match predicted {
                    AlienSex::A => *confusion_matrix_indexing.get_ref_mut([0, 0]) += 1,
                    AlienSex::B => *confusion_matrix_indexing.get_ref_mut([0, 1]) += 1,
                    AlienSex::C => *confusion_matrix_indexing.get_ref_mut([0, 2]) += 1,
                }
            },
            AlienSex::B => {
                match predicted {
                    AlienSex::A => *confusion_matrix_indexing.get_ref_mut([1, 0]) += 1,
                    AlienSex::B => *confusion_matrix_indexing.get_ref_mut([1, 1]) += 1,
                    AlienSex::C => *confusion_matrix_indexing.get_ref_mut([1, 2]) += 1,
                }
            },
            AlienSex::C => {
                match predicted {
                    AlienSex::A => *confusion_matrix_indexing.get_ref_mut([2, 0]) += 1,
                    AlienSex::B => *confusion_matrix_indexing.get_ref_mut([2, 1]) += 1,
                    AlienSex::C => *confusion_matrix_indexing.get_ref_mut([2, 2]) += 1,
                }
            }
        }
    }

    confusion_matrix
};

println!("Confusion matrix: Rows are actual class, Columns are predicted class\n{}",
    confusion_matrix);
println!("  A  B  C");
```

The above code prints 60% accuracy which isn't amazing as if we learned the cluster centers
that labelled the data in the first place we should be able to get 100% accuracy but
60% is still far better than guessing for a three class problem. As Naïve Bayes has no
hyperparameters it is good for establishing a good baseline to compare other examples such
as [Logistic Regression](super::super::logistic_regression) to.
*/

}
