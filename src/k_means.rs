/*!
K-means example

[Overview](https://en.wikipedia.org/wiki/K-means_clustering).

# K means

```
use easy_ml::matrices::Matrix;
use easy_ml::distributions::MultivariateGaussian;

use rand::{Rng, SeedableRng};

/**
 * Utility function to create a list of random numbers.
 */
fn n_random_numbers<R: Rng>(random_generator: &mut R, n: usize) -> Vec<f64> {
    let mut random_numbers = Vec::with_capacity(n);
    for _ in 0..n {
        random_numbers.push(random_generator.gen::<f64>());
    }
    random_numbers
}

// generate two cluster centres using two 2d gaussians, making sure they overlap a bit

// use a fixed seed non cryptographically secure random generator from the rand crate
let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(11);

let cluster1 = MultivariateGaussian::new(
    Matrix::column(vec![ 2.0, 3.0 ]),
    Matrix::from(vec![
        vec![ 1.0, 0.1 ],
        vec![ 0.1, 1.0 ]]));

// make the second cluster more spread out so there will be a bit of overlap with the first
// in the (0,0) to (1, 1) range
let cluster2 = MultivariateGaussian::new(
    Matrix::column(vec![ -2.0, -1.0 ]),
    Matrix::from(vec![
        vec![ 2.5, 1.2 ],
        vec![ 1.2, 2.5 ]]));


let points = 200;
let mut random_numbers = n_random_numbers(&mut random_generator, points * 2);
let cluster1_points = cluster1.draw(&mut random_numbers.drain(..), points).unwrap();
let mut random_numbers = n_random_numbers(&mut random_generator, points * 2);
let cluster2_points = cluster2.draw(&mut random_numbers.drain(..), points).unwrap();

// The following code outputs the data of both clusters in CSV format for
// inspecting in external programs
// TODO: get a scatter plot library
// println!("Cluster 1:");
// for row in 0..cluster1_points.rows() {
//     println!("{},{}", cluster1_points.get(row, 0), cluster1_points.get(row, 1));
// }
//
// println!("Cluster 2:");
// for row in 0..cluster2_points.rows() {
//     println!("{},{}", cluster2_points.get(row, 0), cluster2_points.get(row, 1));
// }


// pick seeds
// extend cluster1_points and cluster2_points to [x, y, cluster]
// reassign clusters & compute centroids
// set some threshold for convergence
```
*/
