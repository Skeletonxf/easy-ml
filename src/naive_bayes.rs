/*!
Naïve Bayes Example

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

## Naïve Bayes Example

```
// For this example some population data is generated for a fictional alien race as I didn't
// have any real datasets to hand
// This alien race has 3 sexes (mysteriously no individuals are ever intersex or trans) and is
// sexually dimorphic, meaning we can try to determine their sex from various measuremnts.

// As with humans, a gaussian distribution for physical charateristics is sensible due to
// evolutionary and biological factors.

use easy_ml::matrices::Matrix;
use easy_ml::matrices::slices::{Slice2D, Slice};
use easy_ml::linear_algebra;
use easy_ml::distributions::Gaussian;

use rand::{Rng, SeedableRng};

struct Alien {
    height: f64,
    primary_marking_color: AlienMarkingColor,
    tail_length: f64,
    metabolic_rate: f64,
    spiked_tail: bool,
    sex: AlienSex,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum AlienMarkingColor {
    Red = 1, Yellow, Orange, Blue, Purple, White, Black
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum AlienSex {
    A, B, C
}

// rather than explicitly define a generative function that already knows the relationship
// between alien charateristics and sex which would be unrealistic in a real example and
// not very interesting, instead the samples are generated without an assigned sex and then
// clustered using k-means

// use a fixed seed non cryptographically secure random generator from the rand crate
let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(16);

struct EndlessRandomGenerator {
    rng: rand_chacha::ChaCha8Rng
}

let mut random_numbers = EndlessRandomGenerator { rng: random_generator };

impl Iterator for EndlessRandomGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.rng.gen::<f64>())
    }
}

fn generate_heights(samples: usize, random_numbers: &mut EndlessRandomGenerator) -> Vec<f64> {
    // the average height shall be 1.5 meters with a standard deviation of 0.25
    let heights_distribution = Gaussian::new(1.5, 0.25 * 0.25);
    let mut heights = heights_distribution.draw(random_numbers, samples).unwrap();
    // ensure all aliens are at least 0.25 meters tall
    heights.drain(..).map(|x| if x > 0.25 { x } else { 0.25 }).collect()
}

fn generate_tail_length(samples: usize, random_numbers: &mut EndlessRandomGenerator) -> Vec<f64> {
    // the average length shall be 1.25 meters with more variation in tail length
    let tails_distribution = Gaussian::new(1.25, 0.5 * 0.5);
    let mut tails = tails_distribution.draw(random_numbers, samples).unwrap();
    // ensure all tails are at least 0.5 meters long
    tails.drain(..).map(|x| if x > 0.5 { x } else { 0.5 }).collect()
}

/*
 * Floats are still returned despite this being a category because we need all the data types
 * to be the same for clustering
 */
fn generate_colors(samples: usize, random_numbers: &mut EndlessRandomGenerator) -> Vec<f64> {
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

fn generate_metabolic_rate(
    samples: usize, random_numbers: &mut EndlessRandomGenerator
) -> Vec<f64> {
    // the average rate shall be 100 heart beats per minute with a standard deviation of 20
    let metabolic_rate_distribution = Gaussian::new(100.0, 20.0 * 20.0);
    let mut metabolic_rates = metabolic_rate_distribution.draw(random_numbers, samples).unwrap();
    // ensure all rates are at least 50 and less than 200
    metabolic_rates.drain(..).map(|x| {
        if x <= 50.0 {
            50.0
        } else if x >= 200.0 {
            200.0
        } else {
            x
        }
    }).collect()
}

fn generate_spiked_tail(
    samples: usize, random_numbers: &mut EndlessRandomGenerator
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

let samples = 1000;

let named_features = vec![ "H", "C", "T", "M", "S" ];

let unlabelled_dataset = {
    let mut dataset = Matrix::column(generate_heights(samples, &mut random_numbers));
    dataset.insert_column_with(1, generate_colors(samples, &mut random_numbers).drain(..));
    dataset.insert_column_with(2, generate_tail_length(samples, &mut random_numbers).drain(..));
    dataset.insert_column_with(3, generate_metabolic_rate(samples, &mut random_numbers).drain(..));
    dataset.insert_column_with(4, generate_spiked_tail(samples, &mut random_numbers).drain(..));
    dataset
};

println!(
    "First 3 rows in unlabeled dataset:\n{:?}",
    unlabelled_dataset.retain(Slice2D::new().columns(Slice::All()).rows(Slice::Range(0..3)))
);

// By clustering only a very small bit of the data, by chance we can expect
// there to be large gaps in the subset because our data is high dimensional
// which will allow k-means to find 3 cluster centres which we will then use
// to assign all the data to classes.

let unlabelled_subset = unlabelled_dataset.retain(
    Slice2D::new()
    .columns(Slice::All())
    .rows(Slice::Range(0..30))
);

// As clustering will use euclidean distance as the distance metric and
// our features have widly varing variances we normalise all the features
// to 0 mean and 1 standard deviation.

let mut means_and_variances = Vec::with_capacity(unlabelled_subset.columns());

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

// create a matrix where each row is the mean and variance of each feature
let means_and_variances = Matrix::from(means_and_variances);

// pick the first 3 samples as the starting points for the cluster centres
// and place them into a 3 x 5 matrix where we have 3 rows of cluster centres
// and 5 features which are all normalised
let mut clusters = Matrix::from(vec![
    normalised_subset.row_iter(0).collect(),
    normalised_subset.row_iter(1).collect(),
    normalised_subset.row_iter(2).collect()]);

// add a 6th column to the points to track the closest cluster
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
            // zip together the index of each point, which will be the
            // row each considered point is stored in
            .zip((0..))
            // exclude the points that aren't assigned to this cluster
            .filter(|(id, index)| (*id as usize) == cluster)
            // drop the cluster ids from each item and copy over the data
            // for each point for each feature
            .map(|(id, index)| {
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

            clusters.set(cluster, feature, new_means[feature]);
        }
    }
}

println!(
    "Denormalised clusters at convergence:\n{:?}\n{:.3}",
    named_features,
    clusters.map_with_index(|x, _, feature| {
        let mean = means_and_variances.get(feature, 0);
        let variance = means_and_variances.get(feature, 1);
        (x * variance) + mean
    }));

//assert_eq!(1, 2);

// TODO:
// Use the 3 cluster centres to assign all the 1000 aliens a sex, creating 3 groups,
// should do this probabalistically to avoid making the problem linearly seperable
// ie if x is 3 times closer to cluster A than B or C then have a 75% chance of assigning it to A
// rather than 100%.
// 2) Construct a vector of aliens and generate some new aliens
// 3) Try to classify the unseen aliens using the 1000 aliens dataset
// Will need both Gaussian naive bayes and laplacian smoothing for categories here
// 4) Finally see what the performance is
```
*/
