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

For more complex data [basis functions](../linear_regression/index.html)
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

Our model **w**<sup>T</sup>**x** is already defined as P(y<sub>i</sub>=True|**x<sub>i</sub>**) so

P(y<sub>i</sub>) = p<sub>i</sub> if y<sub>i</sub> = 1 and 1 - p<sub>i</sub> if y<sub>i</sub> = 0,
where p<sub>i</sub> = **w**<sup>T</sup>**x** = P(y<sub>i</sub>=True|**x<sub>i</sub>**)

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
// the safty of positioning an object on a platform.
// In this example the platform is a rectangle in continuous space. Then we randomly place an
// object somewhere on the platform and apply gaussian noise to it, to model an earthquake or
// strong wind and ask if the object fell off.


use easy_ml::matrices::Matrix;
use easy_ml::matrices::slices::{Slice2D, Slice};
use easy_ml::distributions::MultivariateGaussian;

use rand::{Rng, SeedableRng};

// first we define the AABB bounding box of the platform using a matrix as a datastructure
let platform = Matrix::from(vec![
    vec![ 3.0, 3.0 ],
    vec![ 6.0, 5.0 ]]);

// for the wind we use a simple 0 mean 1 variance multivariate gaussian
let wind = MultivariateGaussian::new(
    Matrix::column(vec![ 0.0, 0.0 ]),
    Matrix::from(vec![
        vec![ 1.0, 0.0 ],
        vec![ 0.0, 1.0 ]]));

const DATASET_SIZE: usize = 100;

// use a fixed seed non cryptographically secure random generator from the rand crate
let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(13);

// generate the dataset of numbers in the range (3, 3) to (6, 4)
let mut objects = Matrix::empty(0.0, (DATASET_SIZE, 2));
for i in 0..DATASET_SIZE {
    // first generate numbers in the [0, 1) range using rand then scale them into
    // the range of the platform
    let position = {
        let x = random_generator.gen::<f64>();
        let y = random_generator.gen::<f64>();
        let x = x * (platform.get(1, 0) - platform.get(0, 0));
        let y = y * (platform.get(1, 1) - platform.get(0, 1));
        let x = x + platform.get(0, 0);
        let y = y + platform.get(0, 1);
        (x, y)
    };
    objects.set(i, 0, position.0);
    objects.set(i, 1, position.1);
}

// insert two new columns into the objects which is their position after being blown by the wind
let blown_objects: Vec<(f64, f64)> = objects.column_iter(0).zip(objects.column_iter(1))
    .map(|(x, y)| {
        // draw a sample for the noise x and y
        let noise = wind.draw(
            &mut vec![ random_generator.gen::<f64>(), random_generator.gen::<f64>() ].drain(..),
            1).unwrap();
        (x + noise.get(0, 0), y + noise.get(0, 1))
    })
    .collect();
objects.insert_column_with(2, blown_objects.iter().map(|&(x, _)| x));
objects.insert_column_with(3, blown_objects.iter().map(|&(_, y)| y));

// check each object's new position and determine if it is still on the platform
// if it's still there we set 1, and if it fell off we set 0.
objects.insert_column_with(4,
    objects.column_iter(2).zip(objects.column_iter(3))
    .map(|(x, y)| {
        if x < platform.get(0, 0) || x > platform.get(1, 0)
                || y < platform.get(0, 1) || y > platform.get(1, 1) {
            0.0
        } else {
            1.0
        }
    })
    // a collect then immediate drain is performed so we stop immutably borrowing
    // from objects before we need to mutably borrow from objects to insert the new column
    .collect::<Vec<f64>>()
    .drain(..)
    );

// objects.insert_column_with(5,
//     objects.column_iter(4)
//     .map(|p| (p / (1.0 - p)).ln())
//     .collect::<Vec<f64>>()
//     .drain(..));

/**
 * The sigmoid function, taking values in the [-inf, inf] range and mapping
 * them into the [0, 1] range
 */
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/**
 * The logit function, taking values in the [0, 1] range and mapping
 * them into the [-inf, inf] range
 */
fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

/**
 * Computes the estimated probability of some feature vector x staying on the
 * platform given the weights in the logistic regression model.
 */
fn probability_estimate(x: &Matrix<f64>, weights: &Matrix<f64>) -> f64 {
    sigmoid((weights.transpose() * x).get(0, 0))
}

/**
 * Computes the negative log likelihood of something? Used to fit the weights for
 * maximum likelihood
 */
fn negative_log_likelihood(x: &Matrix<f64>, weights: &Matrix<f64>, fell: bool) -> f64 {
    if fell {
        // negative case
        -(1.0 - probability_estimate(x, weights).ln())
    } else {
        // positive case
        -(probability_estimate(x, weights).ln())
    }
}

// Plot the first 10 object starting positions and if they fell or not
println!("Objects:\n{:?}", objects.retain(
    Slice2D::new().rows(Slice::Range(0..10)).columns(Slice::Range(0..2).or(Slice::Single(4)))));

// TODO: stochastic gradient descent using the derivative of negative_log_likelihood
// which is: (target - probability_estimate(x, weights)) * x
// ??
// references: http://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf eq 12.12
// https://towardsdatascience.com/understanding-logistic-regression-step-by-step-704a78be7e0a

//assert_eq!(1, 2);
```
*/
