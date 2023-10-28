/*!
K-means example

[Overview](https://en.wikipedia.org/wiki/K-means_clustering).

# K means

The following code creates two 2-dimensional gaussian distributions and then draws samples
from them to create some data which is then assigned to clusters

## Matrix APIs

```
use easy_ml::matrices::Matrix;
use easy_ml::distributions::MultivariateGaussian;

use rand::{Rng, SeedableRng};
use rand::distributions::{DistIter, Standard};
use rand_chacha::ChaCha8Rng;

use rgb::RGB8;
use textplots::{Chart, ColorPlot, Plot, Shape};

// use a fixed seed random generator from the rand crate
let mut random_generator = ChaCha8Rng::seed_from_u64(11);

// define two cluster centres using two 2d gaussians, making sure they overlap a bit
let cluster1 = MultivariateGaussian::new(
    Matrix::column(vec![ 2.0, 3.0 ]),
    Matrix::from(vec![
        vec![ 1.0, 0.1 ],
        vec![ 0.1, 1.0 ]]));

// make the second cluster more spread out so there will be a bit of overlap with the first
// in the (0,0) to (1, 1) area
let cluster2 = MultivariateGaussian::new(
    Matrix::column(vec![ -2.0, -1.0 ]),
    Matrix::from(vec![
        vec![ 2.5, 1.2 ],
        vec![ 1.2, 2.5 ]]));

// Generate 200 points for each cluster
let points = 200;
let mut random_numbers: DistIter<Standard, &mut ChaCha8Rng, f64> =
    (&mut random_generator).sample_iter(Standard);
// we can unwrap here because we deliberately constructed a positive definite covariance matrix
// and supplied enough random numbers
let cluster1_points = cluster1.draw(&mut random_numbers, points).unwrap();
let cluster2_points = cluster2.draw(&mut random_numbers, points).unwrap();

// Plot the generated data into a scatter plot
// There are two clear clusters around the means (of cluster1 and cluster2) but
// many points in the middle are ambiguous, this was deliberate in the choice of
// parameters to generate the data with, as if our data was linearly seperable we
// wouldn't need to perform clustering on it in the first place. Note that, as an unsupervised
// learning method, k-means does not find or try to find a 'right' clustering for arbitary data
println!("Generated data points");
// textplots expects a Vec<(f32, f32)> where each tuple is a (x,y) point to plot,
// so we must transform the data from the cluster points slightly to plot
let scatter_points = cluster1_points.column_iter(0)
    // zip is used to merge the x and y columns in the cluster points into a single tuple
    .zip(cluster1_points.column_iter(1))
    // chain then links the two iterators together so after all of cluster1_points
    // are consumed we use all of cluster2_points
    .chain(cluster2_points.column_iter(0).zip(cluster2_points.column_iter(1)))
    // finally we map the tuples of (f64, f64) into (f32, f32) for handing to the library
    .map(|(x, y)| (x as f32, y as f32))
    .collect::<Vec<(f32, f32)>>();
Chart::new(180, 60, -8.0, 8.0)
    .lineplot(&Shape::Points(&scatter_points))
    .display();


// pick seeds to start each cluster at, in this case we start the seeds at a fixed position
// of (1, 0) and (0, 1) which is deliberately where the two clusters overlap
let mut clusters = Matrix::from(vec![
    vec![ 1.0, 0.0 ],
    vec![ 0.0, 1.0 ]]);

// construct a matrix of rows in the format [x, y, cluster] to contain all the points
let mut points = {
    let mut points = cluster1_points;
    // copy each row of cluster2_points into points
    for row in 0..cluster2_points.rows() {
        // insert each row from cluster2_points to the end of points
        points.insert_row_with(points.rows(), cluster2_points.row_iter(row));
    }
    // extend points from rows of [x, y] to [x, y, cluster] for use in the update loop
    points.insert_column(2, -1.0);
    points
};

// give a name for the meaning of each column in the points matrix
const X: usize = 0;
const Y: usize = 1;
const CLUSTER: usize = 2;

// set a threshold at which we consider the cluster centres to have converged
const CHANGE_THRESHOLD: f64 = 0.001;

// track how much the means have changed each update
let mut absolute_changes = -1.0;

// track where the clusters move over time for plotting
let mut cluster_center_1_history = Vec::with_capacity(7);
let mut cluster_center_2_history = Vec::with_capacity(7);

// loop until we go under the CHANGE_THRESHOLD, reassigning points to the nearest
// cluster then cluster centres to their mean of points
while absolute_changes == -1.0 || absolute_changes > CHANGE_THRESHOLD {
    println!("Cluster centres: ({},{}), ({},{})",
        clusters.get(0, X), clusters.get(0, Y),
        clusters.get(1, X), clusters.get(1, Y));
    cluster_center_1_history.push((clusters.get(0, X) as f32, clusters.get(0, Y) as f32));
    cluster_center_2_history.push((clusters.get(1, X) as f32, clusters.get(1, Y) as f32));

    // assign each point to the nearest cluster centre by euclidean distance
    for point in 0..points.rows() {
        let x = points.get(point, X);
        let y = points.get(point, Y);
        let mut closest_cluster = -1.0;
        let mut least_squared_distance = std::f64::MAX;
        for cluster in 0..clusters.rows() {
            let cx = clusters.get(cluster, X);
            let cy = clusters.get(cluster, Y);
            // we don't actually need to square root the distances for finding
            // which is least because least squared distance is the same as
            // least distance
            let squared_distance = (x - cx).powi(2) + (y - cy).powi(2);
            if squared_distance < least_squared_distance {
                closest_cluster = cluster as f64;
                least_squared_distance = squared_distance;
            }
        }
        // save the cluster that is closest to each point
        points.set(point, CLUSTER, closest_cluster);
    }
    // update cluster centres to the mean of their points
    absolute_changes = 0.0;
    for cluster in 0..clusters.rows() {
        // construct a list of the points this cluster owns
        let owned = points.column_iter(CLUSTER)
            // zip together the cluster id in each point with their X, Y points
            .zip(points.column_reference_iter(X).zip(points.column_reference_iter(Y)))
            // exclude the points that aren't assigned to this cluster
            .filter(|(id, (x, y))| (*id as usize) == cluster)
            // drop the cluster ids from each item
            .map(|(id, (x, y))| (x, y))
            // collect into a vector of tuples
            .collect::<Vec<(&f64, &f64)>>();
        let total = owned.len() as f64;
        let mean_x = owned.iter().map(|(&x, _)| x).sum::<f64>() / total;
        let mean_y = owned.iter().map(|(_, &y)| y).sum::<f64>() / total;
        // track the absolute difference between the new mean and the old one
        // so we know when to stop updating the clusters
        absolute_changes += (clusters.get(cluster, X) - mean_x).abs();
        absolute_changes += (clusters.get(cluster, Y) - mean_y).abs();
        // set the new mean x and y for this cluster
        clusters.set(cluster, X, mean_x);
        clusters.set(cluster, Y, mean_y);
    }
}
println!("Cluster centres: ({},{}), ({},{})",
    clusters.get(0, X), clusters.get(0, Y),
    clusters.get(1, X), clusters.get(1, Y));
cluster_center_1_history.push((clusters.get(0, X) as f32, clusters.get(0, Y) as f32));
cluster_center_2_history.push((clusters.get(1, X) as f32, clusters.get(1, Y) as f32));

println!("Cluster centre movements");
Chart::new(180, 60, -8.0, 8.0)
    .lineplot(&Shape::Points(&scatter_points))
    .linecolorplot(&Shape::Lines(&cluster_center_1_history), RGB8::new(255, 100, 100))
    .linecolorplot(&Shape::Lines(&cluster_center_2_history), RGB8::new(100, 100, 255))
    .display();
```

## Tensor APIs

```
use easy_ml::tensors::Tensor;
use easy_ml::tensors::views::TensorStack;
use easy_ml::distributions::MultivariateGaussianTensor;

use rand::{Rng, SeedableRng};
use rand::distributions::{DistIter, Standard};
use rand_chacha::ChaCha8Rng;

use rgb::RGB8;
use textplots::{Chart, ColorPlot, Plot, Shape};

// use a fixed seed random generator from the rand crate
let mut random_generator = ChaCha8Rng::seed_from_u64(11);

// define two cluster centres using two 2d gaussians, making sure they overlap a bit
let cluster1 = MultivariateGaussianTensor::new(
    Tensor::from([("means", 2)], vec![ 2.0, 3.0 ]),
    Tensor::from(
        [("rows", 2), ("columns", 2)],
        vec![
            1.0, 0.1,
            0.1, 1.0
        ]
    )
).unwrap(); // we can unwrap here because we know we supplied valid inputs to the Gaussian

// make the second cluster more spread out so there will be a bit of overlap with the first
// in the (0,0) to (1, 1) area
let cluster2 = MultivariateGaussianTensor::new(
    Tensor::from([("means", 2)], vec![ -2.0, -1.0 ]),
    Tensor::from(
        [("rows", 2), ("columns", 2)],
        vec![
            2.5, 1.2,
            1.2, 2.5
        ]
    )
).unwrap(); // we can unwrap here because we know we supplied valid inputs to the Gaussian

// Generate 200 points for each cluster
let points = 200;
let mut random_numbers: DistIter<Standard, &mut ChaCha8Rng, f64> =
    (&mut random_generator).sample_iter(Standard);
// we can unwrap here because we deliberately constructed a positive definite covariance matrix
// and supplied enough random numbers
let cluster1_points = cluster1.draw(&mut random_numbers, points, "data", "feature").unwrap();
let cluster2_points = cluster2.draw(&mut random_numbers, points, "data", "feature").unwrap();

// Plot the generated data into a scatter plot
// There are two clear clusters around the means (of cluster1 and cluster2) but
// many points in the middle are ambiguous, this was deliberate in the choice of
// parameters to generate the data with, as if our data was linearly seperable we
// wouldn't need to perform clustering on it in the first place. Note that, as an unsupervised
// learning method, k-means does not find or try to find a 'right' clustering for arbitary data
println!("Generated data points");
// textplots expects a Vec<(f32, f32)> where each tuple is a (x,y) point to plot,
// so we must transform the data from the cluster points slightly to plot
let scatter_points = cluster1_points
    .select([("feature", 0)])
    .iter()
    // zip is used to merge the x and y columns in the cluster points into a single tuple
    .zip(cluster1_points.select([("feature", 1)]).iter())
    // chain then links the two iterators together so after all of cluster1_points
    // are consumed we use all of cluster2_points
    .chain(
        cluster2_points
            .select([("feature", 0)])
            .iter()
            .zip(cluster2_points.select([("feature", 1)]).iter())
    )
    // finally we map the tuples of (f64, f64) into (f32, f32) for handing to the library
    .map(|(x, y)| (x as f32, y as f32))
    .collect::<Vec<(f32, f32)>>();

Chart::new(180, 60, -8.0, 8.0)
    .lineplot(&Shape::Points(&scatter_points))
    .display();


// pick seeds to start each cluster at, in this case we start the seeds at a fixed position
// of (1, 0) and (0, 1) which is deliberately where the two clusters overlap
let mut clusters = Tensor::from(
    [("cluster", 2), ("xy", 2)],
    vec![
        1.0, 0.0,
        0.0, 1.0
    ]
);

// construct a matrix of rows in the format [x, y, cluster] to contain all the points
let mut points = {
    let mut points = Tensor::empty(
        [("data", 400), ("feature", 3)],
        -1.0
    );
    // copy in the rows of cluster1_points and cluster2_points
    let mut data = cluster1_points.iter().chain(cluster2_points.iter());
    for ([_row, feature], x) in points.iter_reference_mut().with_index() {
        *x = match feature {
            // x and y come from cluster points
            0 | 1 => data.next().unwrap(),
            _ => -1.0,
        };
    }
    points
};

// give a name for the meaning of each feature in the points matrix
const X: usize = 0;
const Y: usize = 1;
const CLUSTER: usize = 2;

// set a threshold at which we consider the cluster centres to have converged
const CHANGE_THRESHOLD: f64 = 0.001;

// track how much the means have changed each update
let mut absolute_changes = -1.0;

// track where the clusters move over time for plotting
let mut cluster_center_1_history = Vec::with_capacity(7);
let mut cluster_center_2_history = Vec::with_capacity(7);

// loop until we go under the CHANGE_THRESHOLD, reassigning points to the nearest
// cluster then cluster centres to their mean of points
while absolute_changes == -1.0 || absolute_changes > CHANGE_THRESHOLD {
    let mut clusters = clusters.index_by_mut(["cluster", "xy"]);
    println!("Cluster centres: ({},{}), ({},{})",
        clusters.get([0, X]), clusters.get([0, Y]),
        clusters.get([1, X]), clusters.get([1, Y])
    );
    cluster_center_1_history.push((clusters.get([0, X]) as f32, clusters.get([0, Y]) as f32));
    cluster_center_2_history.push((clusters.get([1, X]) as f32, clusters.get([1, Y]) as f32));

    let number_of_points = points.shape()[0].1;
    let number_of_clusters = clusters.shape()[0].1;
    // assign each point to the nearest cluster centre by euclidean distance
    {
        let mut points = points.index_by_mut(["data", "feature"]);
        for point in 0..number_of_points {
            let x = points.get([point, X]);
            let y = points.get([point, Y]);
            let mut closest_cluster = -1.0;
            let mut least_squared_distance = std::f64::MAX;
            for cluster in 0..number_of_clusters {
                let cx = clusters.get([cluster, X]);
                let cy = clusters.get([cluster, Y]);
                // we don't actually need to square root the distances for finding
                // which is least because least squared distance is the same as
                // least distance
                let squared_distance = (x - cx).powi(2) + (y - cy).powi(2);
                if squared_distance < least_squared_distance {
                    closest_cluster = cluster as f64;
                    least_squared_distance = squared_distance;
                }
            }
            // save the cluster that is closest to each point
            *points.get_ref_mut([point, CLUSTER]) = closest_cluster;
        }
    } // drop the TensorAccess wrapper on points

    // update cluster centres to the mean of their points
    absolute_changes = 0.0;
    for cluster in 0..number_of_clusters {
        // construct a list of the points this cluster owns
        let owned = points.select([("feature", CLUSTER)]).iter()
            // zip together the cluster id in each point with their X, Y points
            .zip(
                points.select([("feature", X)]).iter()
                    .zip(points.select([("feature", Y)]).iter())
            )
            // exclude the points that aren't assigned to this cluster
            .filter(|(id, (x, y))| (*id as usize) == cluster)
            // drop the cluster ids from each item
            .map(|(id, (x, y))| (x, y))
            // collect into a vector of tuples
            .collect::<Vec<(f64, f64)>>();
        let total = owned.len() as f64;
        let mean_x = owned.iter().map(|(x, _)| x).sum::<f64>() / total;
        let mean_y = owned.iter().map(|(_, y)| y).sum::<f64>() / total;
        // track the absolute difference between the new mean and the old one
        // so we know when to stop updating the clusters
        absolute_changes += (clusters.get([cluster, X]) - mean_x).abs();
        absolute_changes += (clusters.get([cluster, Y]) - mean_y).abs();
        // set the new mean x and y for this cluster
        *clusters.get_ref_mut([cluster, X]) = mean_x;
        *clusters.get_ref_mut([cluster, Y]) = mean_y;
    }
}
let clusters = clusters.index_by(["cluster", "xy"]);
println!("Cluster centres: ({},{}), ({},{})",
    clusters.get([0, X]), clusters.get([0, Y]),
    clusters.get([1, X]), clusters.get([1, Y]));
cluster_center_1_history.push((clusters.get([0, X]) as f32, clusters.get([0, Y]) as f32));
cluster_center_2_history.push((clusters.get([1, X]) as f32, clusters.get([1, Y]) as f32));

println!("Cluster centre movements");
Chart::new(180, 60, -8.0, 8.0)
    .lineplot(&Shape::Points(&scatter_points))
    .linecolorplot(&Shape::Lines(&cluster_center_1_history), RGB8::new(255, 100, 100))
    .linecolorplot(&Shape::Lines(&cluster_center_2_history), RGB8::new(100, 100, 255))
    .display();
```

# 5 Dimensional K-means

See [naive_bayes](super::naive_bayes::three_class)
*/
