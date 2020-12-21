/*!
Logistic regression example

Logistic regression can be used for classification. By performing linear regression on a logit
function a linear classifier can be obtained that retains probabilistic semantics.

Given some data on a binary classification problem (ie a
[Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)),
transforming the probabilities with a logit function:

<pre>log(p / (1 - p))</pre>

where p is the probability of success, ie P(y=True|x), puts them in the -infinity to infinity
range. If we assume a simple linear model over two inputs x1 and x2 then:

<pre>log(p / (1 - p)) = w0 + w1*x1 + w2*x2</pre>

For more complex data [basis functions](super::linear_regression)
can be used on the inputs to model non linearity. Once we have a model we can define the
objective function to maximise to learn the weights for. Once we have fixed weights
we can estimate the probability of new data by taking the inverse of the logit function
(the sigmoid function):

<pre>1 / (1 + e^(-(w0 + w1*x1 + w2*x2)))</pre>

which maps back into the 0 - 1 range and produces a probability for the unseen data. We can then
choose a cutoff at say 0.5 and we have a classifier that ouputs True for any unseen data estimated
to have a probability &ge; 0.5 and False otherwise.

# Arriving at the update rule

If the samples are independent of each other, ie knowing P(y<sub>1</sub>=True|x<sub>1</sub>)
tells you nothing about P(y<sub>1</sub>=True|x<sub>2</sub>), as is the case in a bernoulli
distribution, then the probability of P(**y**|X) is the product of each
P(y<sub>i</sub>|**x<sub>i</sub>**). For numerical stability reasons we often want to take logs
of the probability, which transforms the product into a sum.

log(P(**y**|X)) = the sum over all i data of (log(P(y<sub>i</sub>|**x<sub>i</sub>**)))

Our model sigmoid(**w**<sup>T</sup>**x**) is already defined as P(y<sub>i</sub>=True|**x<sub>i</sub>**) so

P(y<sub>i</sub>) = p<sub>i</sub> if y<sub>i</sub> = 1 and 1 - p<sub>i</sub> if y<sub>i</sub> = 0,
where p<sub>i</sub> = P(y<sub>i</sub>=True|**x<sub>i</sub>**) = sigmoid(**w**<sup>T</sup>**x**)

this can be converted into a single equation because a<sup>0</sup> = 1

P(y<sub>i</sub>) = (p<sub>i</sub>^y<sub>i</sub>) * ((1 - p<sub>i</sub>)^(1 - y<sub>i</sub>))

putting the two equations together gives the log probability we want to maximise in terms of
p<sub>i</sub>, which is itself in terms of our model's weights

log(P(**y**|X)) = the sum over all i data of (log((p<sub>i</sub>^y<sub>i</sub>) * (1 - p<sub>i</sub>)^(1 - y<sub>i</sub>)))

by log rules we can remove the exponents

log(P(**y**|X)) = the sum over all i data of (y<sub>i</sub> * log(p<sub>i</sub>) + (1 - y<sub>i</sub>) * log(1 - p<sub>i</sub>)))

we want to maximise P(**y**|X) with our weights so we take the derivative with respect to **w**

d(log(P(**y**|X))) / d**w** = the sum over all i data of ((y<sub>i</sub> - p(y<sub>i</sub>=True|**x<sub>i</sub>**))**x<sub>i</sub>**)

where p(y<sub>i</sub>=True|**x<sub>i</sub>**) = 1 / (1 + e^(-(w0 + w1 * x1 + w2 * x2))) as defined
earlier. This derivative will maximise log(P(**y**|X)), and as logs are monotonic P(**y**|X) as
well, when it equals 0. Unfortunatly there is no closed form solution so we must perform gradient
descent to fit **w**. In this example i is small enough we perform gradient descent over all the
training data, for big data problems stochastic gradient descent would scale better.

The update rule:

**w<sub>new</sub>** = **w<sub>old</sub>** + learning_rate * (the sum over all i data of (y<sub>i</sub> - [1 / (1 + e^(-(**w**<sup>T</sup>**x**)))])**x<sub>i</sub>**))

# Logistic regression example
```
// Actual types of datasets logistic regression might be performed on include diagnostic
// datasets such as cancer/not cancer diagnosis and various measurements of patients.
// More abstract datasets could be related to coin flipping.
//
// To ensure our example is not overly complicated but requires two dimensions for the inputs
// to estimate the probability distribution of the random variable we sample from, we model
// two clusters from different classes to classify. As each class is arbitary, we assign the first
// class as the True case that the model should predict >0.5 probability for, and the second
// class as the False case that the model should predict <0.5 probability for.

use easy_ml::matrices::Matrix;
use easy_ml::matrices::slices::{Slice2D, Slice};
use easy_ml::distributions::MultivariateGaussian;

use rand::{Rng, SeedableRng};

use textplots::{Chart, Plot, Shape};

// use a fixed seed non cryptographically secure random generator from the rand crate
let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(13);

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

// define two cluster centres using two 2d gaussians, making sure they overlap a bit
let class1 = MultivariateGaussian::new(
    Matrix::column(vec![ 2.0, 3.0 ]),
    Matrix::from(vec![
        vec![ 1.0, 0.1 ],
        vec![ 0.1, 1.0 ]]));

// make the second cluster more spread out so there will be a bit of overlap with the first
// in the (0,0) to (1, 1) area
let class2 = MultivariateGaussian::new(
    Matrix::column(vec![ -2.0, -1.0 ]),
    Matrix::from(vec![
        vec![ 2.5, 1.2 ],
        vec![ 1.2, 2.5 ]]));

// Generate 200 points for each cluster
let points = 200;
let mut random_numbers = n_random_numbers(&mut random_generator, points * 2);
let class1_points = class1.draw(&mut random_numbers.drain(..), points).unwrap();
let mut random_numbers = n_random_numbers(&mut random_generator, points * 2);
let class2_points = class2.draw(&mut random_numbers.drain(..), points).unwrap();

// Plot each class of the generated data in a scatter plot
println!("Generated data points");

/**
 * Helper function to print a scatter plot of a provided matrix with x, y in each row
 */
fn scatter_plot(data: &Matrix<f64>) {
    // textplots expects a Vec<(f32, f32)> where each tuple is a (x,y) point to plot,
    // so we must transform the data from the cluster points slightly to plot
    let scatter_points = data.column_iter(0)
        // zip is used to merge the x and y columns in the data into a single tuple
        .zip(data.column_iter(1))
        // finally we map the tuples of (f64, f64) into (f32, f32) for handing to the library
        .map(|(x, y)| (x as f32, y as f32))
        .collect::<Vec<(f32, f32)>>();
    Chart::new(180, 60, -8.0, 8.0)
        .lineplot(Shape::Points(&scatter_points))
        .display();
}

println!("Classs 1");
scatter_plot(&class1_points);
println!("Classs 2");
scatter_plot(&class2_points);

// for ease of use later we insert a 0th column into both class's points so w0 + w1*x1 + w2*x2
// can be computed by w^T x
let class1_inputs = {
    let mut design_matrix = class1_points.clone();
    design_matrix.insert_column(0, 1.0);
    design_matrix
};
let class2_inputs =  {
    let mut design_matrix = class2_points.clone();
    design_matrix.insert_column(0, 1.0);
    design_matrix
};

/**
 * The sigmoid function, taking values in the [-inf, inf] range and mapping
 * them into the [0, 1] range.
 */
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + ((-x).exp()))
}

/**
 * The logit function, taking values in the [0, 1] range and mapping
 * them into the [-inf, inf] range
 */
fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

// First we initialise the weights matrix to some initial values
let mut weights = Matrix::column(vec![ 1.0, 1.0, 1.0 ]);

/**
 * The log of the likelihood function P(**y**|X). This is what we want to update our
 * weights to maximise as we want to train the model to predict y given **x**,
 * where y is the class and **x** is the two features the model takes as input.
 * It should be noted that something has probably gone wrong if you ever get 100%
 * performance on your training data, either your training data is linearly seperable
 * or you are overfitting and the weights won't generalise to predicting the correct
 * class given unseen inputs.
 *
 * This function is mostly defined for completeness, we maximise it using the derivative
 * and never need to compute it.
 */
fn log_likelihood(
    weights: &Matrix<f64>, class1_inputs: &Matrix<f64>, class2_inputs: &Matrix<f64>
) -> f64 {
    // The probability of predicting all inputs as the correct class is the product
    // of the probability of predicting each individal input and class correctly, as we
    // assume each sample is independent of each other.
    let mut likelihood = 1_f64.ln();
    // the model should predict 1 for each class 1
    let predictions = (class1_inputs * weights).map(sigmoid);
    for i in 0..predictions.rows() {
        likelihood += predictions.get(i, 0).ln();
    }
    // the model should predict 0 for each class 2
    let predictions = (class2_inputs * weights).map(sigmoid).map(|p| 1.0 - p);
    for i in 0..predictions.rows() {
        likelihood += predictions.get(i, 0).ln();
    }
    likelihood
}

/**
 * The derivative of the negative log likelihood function, which we want to set to 0 in order
 * to maximise P(**y**|X).
 */
fn update_function(
    weights: &Matrix<f64>, class1_inputs: &Matrix<f64>, class2_inputs: &Matrix<f64>
) -> Matrix<f64> {
    let mut derivative = Matrix::column(vec![ 0.0, 0.0, 0.0 ]);

    // compute y - predictions for all the first class of inputs
    let prediction_errors = (class1_inputs * weights).map(sigmoid).map(|p| 1.0 - p);
    for i in 0..prediction_errors.rows() {
        // compute diff * x_i
        let diff = prediction_errors.get(i, 0);
        let ith_error = Matrix::column(class1_inputs.row_iter(i).collect()).map(|x| x * diff);
        derivative = derivative + ith_error;
    }

    // compute y - predictions for all the second class of inputs
    let prediction_errors = (class2_inputs * weights).map(sigmoid).map(|p| 0.0 - p);
    for i in 0..prediction_errors.rows() {
        // compute diff * x_i
        let diff = prediction_errors.get(i, 0);
        let ith_error = Matrix::column(class2_inputs.row_iter(i).collect()) * diff;
        derivative = derivative + ith_error;
    }

    derivative
}

let learning_rate = 0.002;

let mut log_likelihood_progress = Vec::with_capacity(25);

// For this example we cheat and have simply found what number of iterations and learning rate
// yields a correct decision boundry so don't actually check for convergence. In a real example
// you would stop once the updates for the weights become 0 or very close to 0.
for i in 0..25 {
    let update = update_function(&weights, &class1_inputs, &class2_inputs);
    weights = weights + (update * learning_rate);
    log_likelihood_progress.push(
        (i as f32, log_likelihood(&weights, &class1_inputs, &class2_inputs) as f32)
    );
}

println!("Log likelihood over 25 iterations (bigger is better as logs are monotonic)");
Chart::new(180, 60, 0.0, 15.0)
    .lineplot(Shape::Lines(&log_likelihood_progress))
    .display();

println!("Decision boundry after 25 iterations");
decision_boundry(&weights);

// The model should have learnt to classify class 1 correctly at the expected value
// of the cluster
assert!(
    sigmoid(
        (weights.transpose() * Matrix::column(vec![ 1.0, 2.0, 3.0])).scalar()
    ) > 0.5);

// The model should have learnt to classify class 2 correctly at the expected value
// of the cluster
assert!(
    sigmoid(
        (weights.transpose() * Matrix::column(vec![ 1.0, -2.0, -1.0])).scalar()
    ) < 0.5);

/**
 * A utility function to plot the decision boundry of the model. As the terminal plotting
 * library doesn't support colored plotting at the time of writing this is a little challenging
 * to do given we have two dimensions of inputs and one dimension of output which is also real
 * valued as logistic regression computes probability. This could best be done with a 3d
 * plot or a heatmap, but is done with this function by taking 0.5 as the cutoff for
 * classification, generating a grid of points in the two dimensional space and classifying all
 * of them, then plotting the ones classified as class 1.
 *
 * TODO: work out how to retrieve a line from these points so the decision boundry is easier
 * to see
 */
fn decision_boundry(weights: &Matrix<f64>) {
    // compute a matrix of coordinate pairs from (-8.0, -8.0) to (8.0, 8.0)
    let grid_values = Matrix::empty(0.0, (160, 160));
    // create a matrix of tuples combining every combination of coordinates
    let grid_values = grid_values.map_with_index(|_, i, j| (
        (i as f64 - 80.0) * 0.1, (j as f64 - 80.0) * 0.1)
    );
    // iterate through every tuple and see if the model predicts class 1
    let points = grid_values.column_major_iter()
        .map(|(x1, x2)| {
            let input = Matrix::column(vec![ 1.0, x1, x2 ]);
            let prediction = sigmoid((weights.transpose() * input).scalar());
            return if prediction > 0.5 {
                (x1, x2, 1)
            } else {
                (x1, x2, 0)
            }
        })
        .filter(|(_, _, class)| class == &1)
        .map(|(x1, x2, _)| (x1 as f32, x2 as f32))
        .collect::<Vec<(f32, f32)>>();
    Chart::new(180, 60, -8.0, 8.0)
        .lineplot(Shape::Points(&points))
        .display();
}
```
*/
