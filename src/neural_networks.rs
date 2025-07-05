/*!
Neural Network training examples

# XOR

The following code shows a simple network using the sigmoid activation function
to learn the non linear XOR function. Use of a non linear activation function
is very important, as without them the network would not be able to remap the
inputs into a new space that can be linearly seperated.

Rather than symbolically differentiate the model y = sigmoid(sigmoid(x * w1) * w2) * w3
the [Record](super::differentiation::Record) struct is used to perform reverse
[automatic differentiation](super::differentiation). This adds a slight
memory overhead but also makes it easy to experiment with adding or tweaking layers
of the network or trying different activation functions like ReLu or tanh.

Note that the gradients recorded in each epoch must be cleared before training in
the next one.

## Matrix APIs

```
use easy_ml::matrices::Matrix;
use easy_ml::matrices::views::{MatrixRange, MatrixView, MatrixRef, NoInteriorMutability, IndexRange};
use easy_ml::numeric::Numeric;
use easy_ml::numeric::extra::Real;
use easy_ml::differentiation::{Record, RecordMatrix, WengertList, Index};

use rand::{Rng, SeedableRng};
use rand::distr::StandardUniform;

use textplots::{Chart, Plot, Shape};

/**
 * Utility function to create a list of random numbers.
 */
fn n_random_numbers<R: Rng>(random_generator: &mut R, n: usize) -> Vec<f32> {
    random_generator.sample_iter(StandardUniform).take(n).collect()
}

/**
 * The sigmoid function which will be used as a non linear activation function.
 *
 * This is written for a generic type, so it can be used with records and also
 * with normal floats.
 */
fn sigmoid<T: Real + Copy>(x: T) -> T {
    // 1 / (1 + e^-x)
    T::one() / (T::one() + (-x).exp())
}

/**
 * A simple three layer neural network that outputs a scalar.
 */
fn model(
    input: &Matrix<f32>, w1: &Matrix<f32>, w2: &Matrix<f32>, w3: &Matrix<f32>
) -> f32 {
    (((input * w1).map(sigmoid) * w2).map(sigmoid) * w3).scalar()
}

/**
 * A simple three layer neural network that outputs a scalar, using RecordMatrix types for the
 * inputs to track derivatives.
 */
fn model_training<'a, I>(
    input: &RecordMatrix<'a, f32, I>,
    w1: &RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
    w2: &RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
    w3: &RecordMatrix<'a, f32, Matrix<(f32, Index)>>
) -> Record<'a, f32>
where
    I: MatrixRef<(f32, Index)> + NoInteriorMutability,
{
    (((input * w1).map(sigmoid).unwrap() * w2).map(sigmoid).unwrap() * w3).get_as_record(0, 0)
}

/**
 * Computes mean squared loss of the network against all the training data.
 */
fn mean_squared_loss(
   inputs: &Vec<Matrix<f32>>,
   w1: &Matrix<f32>,
   w2: &Matrix<f32>,
   w3: &Matrix<f32>,
   labels: &Vec<f32>
) -> f32 {
    inputs.iter().enumerate().fold(0.0, |acc, (i, input)| {
        let output = model(input, w1, w2, w3);
        let correct = labels[i];
        // sum up the squared loss
        acc + ((correct - output) * (correct - output))
    }) / inputs.len() as f32
}

/**
 * Computes mean squared loss of the network against all the training data, using RecordMatrix
 * types for the inputs to track derivatives.
 */
fn mean_squared_loss_training<'a>(
    inputs: &RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
    w1: &RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
    w2: &RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
    w3: &RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
    labels: &RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
) -> Record<'a, f32> {
    let rows = inputs.rows();
    let columns = inputs.columns();
    let history = w1.history();
    (0..rows).map(|r| {
        // take each row as its own RecordMatrix input
        (r, RecordMatrix::from_existing(
            history,
            MatrixView::from(
                MatrixRange::from(
                    inputs,
                    IndexRange::new(r, 1),
                    IndexRange::new(0, columns),
                )
            )
        ))
    }).fold(Record::constant(0.0), |acc, (r, input)| {
        let output = model_training(&input, w1, w2, w3);
        let correct = labels.get_as_record(0, r);
        // sum up the squared loss
        acc + ((correct - output) * (correct - output))
    }) / (rows as f32)
}

/**
 * Updates the weight matrices to step the gradient by one step.
 *
 * Note that here we need the methods defined on Record / RecordMatrix to do backprop. There is
 * no non-training version of this we can define without deriative tracking.
 */
fn step_gradient<'a>(
    inputs: &RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
    w1: &mut RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
    w2: &mut RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
    w3: &mut RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
    labels: &RecordMatrix<'a, f32, Matrix<(f32, Index)>>,
    learning_rate: f32,
    list: &'a WengertList<f32>
) -> f32 {
    let loss = mean_squared_loss_training(inputs, w1, w2, w3, labels);
    let derivatives = loss.derivatives();
    // update each element in the weight matrices by the derivatives
    // unwrapping here is fine because we're not changing the history of any variables
    w1.map_mut(|x| x - (derivatives[&x] * learning_rate)).unwrap();
    w2.map_mut(|x| x - (derivatives[&x] * learning_rate)).unwrap();
    w3.map_mut(|x| x - (derivatives[&x] * learning_rate)).unwrap();
    // reset gradients
    list.clear();
    w1.reset();
    w2.reset();
    w3.reset();
    // return the loss
    loss.number
}

// use a fixed seed random generator from the rand crate
let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(25);

// randomly initalise the weights using the fixed seed generator for reproducibility
let list = WengertList::new();
// w1 will be a 3x3 matrix
let mut w1 = RecordMatrix::variables(
    &list,
    Matrix::from(
        vec![
            n_random_numbers(&mut random_generator, 3),
            n_random_numbers(&mut random_generator, 3),
            n_random_numbers(&mut random_generator, 3)
        ]
    )
);
// w2 will be a 3x3 matrix
let mut w2 = RecordMatrix::variables(
    &list,
    Matrix::from(
        vec![
            n_random_numbers(&mut random_generator, 3),
            n_random_numbers(&mut random_generator, 3),
            n_random_numbers(&mut random_generator, 3)
        ]
    )
);
// w3 will be a 3x1 column matrix
let mut w3 = RecordMatrix::variables(
    &list,
    Matrix::column(n_random_numbers(&mut random_generator, 3))
);
println!("w1 {}", w1);
println!("w2 {}", w2);
println!("w3 {}", w3);

// define XOR inputs, with biases added to the inputs
let inputs = RecordMatrix::constants(
    Matrix::from(
        vec![
            vec![ 0.0, 0.0, 1.0 ],
            vec![ 0.0, 1.0, 1.0 ],
            vec![ 1.0, 0.0, 1.0 ],
            vec![ 1.0, 1.0, 1.0 ],
        ]
    )
);
// define XOR outputs which will be used as labels
let labels = RecordMatrix::constants(
    Matrix::row(vec![ 0.0, 1.0, 1.0, 0.0 ])
);
let learning_rate = 0.2;
let epochs = 4000;

// do the gradient descent and save the loss at each epoch
let mut losses = Vec::with_capacity(epochs);
for _ in 0..epochs {
    losses.push(step_gradient(&inputs, &mut w1, &mut w2, &mut w3, &labels, learning_rate, &list))
}

// now plot the training loss
let mut chart = Chart::new(180, 60, 0.0, epochs as f32);
chart.lineplot(
    &Shape::Lines(&losses.iter()
        .cloned()
        .enumerate()
        .map(|(i, x)| (i as f32, x))
        .collect::<Vec<(f32, f32)>>())
    ).display();

// note that with different hyperparameters, starting weights, or less training
// the network may not have converged and could still be outputting 0.5 for everything,
// the chart plot with this configuration is particularly interesting because the loss
// hovers around 0.3 to 0.2 for a while (while outputting 0.5 for every input) before
// finally learning how to remap the input data in a way which can then be linearly
// seperated to achieve ~0.0 loss.

// check that the weights are sensible
println!("w1 {}", w1);
println!("w2 {}", w2);
println!("w3 {}", w3);
// check that the network has learned XOR properly

let row_1 = RecordMatrix::from_existing(
    Some(&list),
    MatrixView::from(MatrixRange::from(&inputs, 0..1, 0..3))
);
let row_2 = RecordMatrix::from_existing(
    Some(&list),
    MatrixView::from(MatrixRange::from(&inputs, 1..2, 0..3))
);
let row_3 = RecordMatrix::from_existing(
    Some(&list),
    MatrixView::from(MatrixRange::from(&inputs, 2..3, 0..3))
);
let row_4 = RecordMatrix::from_existing(
    Some(&list),
    MatrixView::from(MatrixRange::from(&inputs, 3..4, 0..3))
);
println!("0 0: {:?}", model_training(&row_1, &w1, &w2, &w3).number);
println!("0 1: {:?}", model_training(&row_2, &w1, &w2, &w3).number);
println!("1 0: {:?}", model_training(&row_3, &w1, &w2, &w3).number);
println!("1 1: {:?}", model_training(&row_4, &w1, &w2, &w3).number);
assert!(losses[epochs - 1] < 0.02);

// we can also extract the learned weights once done with training and avoid the memory
// overhead of Record
let w1_final = w1.view().map(|(x, _)| x);
let w2_final = w2.view().map(|(x, _)| x);
let w3_final = w3.view().map(|(x, _)| x);
println!("0 0: {:?}", model(&row_1.view().map(|(x, _)| x), &w1_final, &w2_final, &w3_final));
println!("0 1: {:?}", model(&row_2.view().map(|(x, _)| x), &w1_final, &w2_final, &w3_final));
println!("1 0: {:?}", model(&row_3.view().map(|(x, _)| x), &w1_final, &w2_final, &w3_final));
println!("1 1: {:?}", model(&row_4.view().map(|(x, _)| x), &w1_final, &w2_final, &w3_final));
```

## Tensor APIs

```
use easy_ml::tensors::Tensor;
use easy_ml::tensors::views::{TensorView, TensorRef};
use easy_ml::numeric::{Numeric, NumericRef};
use easy_ml::numeric::extra::{Real, RealRef, Exp};
use easy_ml::differentiation::{Record, RecordTensor, WengertList, Index};

use rand::{Rng, SeedableRng};
use rand::distr::StandardUniform;

use textplots::{Chart, Plot, Shape};

/**
 * Utility function to create a list of random numbers.
 */
fn n_random_numbers<R: Rng>(random_generator: &mut R, n: usize) -> Vec<f32> {
    random_generator.sample_iter(StandardUniform).take(n).collect()
}

/**
 * The sigmoid function which will be used as a non linear activation function.
 *
 * This is written for a generic type, so it can be used with records and also
 * with normal floats.
 */
fn sigmoid<T: Real + Copy>(x: T) -> T {
    // 1 / (1 + e^-x)
    T::one() / (T::one() + (-x).exp())
}

/**
 * A simple three layer neural network that outputs a scalar.
 */
fn model<I>(
    input: &TensorView<f32, I, 2>,
    w1: &Tensor<f32, 2>,
    w2: &Tensor<f32, 2>,
    w3: &Tensor<f32, 2>,
) -> f32
where
    I: TensorRef<f32, 2>,
{
    (((input * w1).map(sigmoid) * w2).map(sigmoid) * w3).first()
}

/**
 * A simple three layer neural network that outputs a scalar, using RecordTensor types for
 * the inputs to track derivatives.
 */
fn model_training<'a, I>(
    input: &RecordTensor<'a, f32, I, 2>,
    w1: &RecordTensor<'a, f32, Tensor<(f32, Index), 2>, 2>,
    w2: &RecordTensor<'a, f32, Tensor<(f32, Index), 2>, 2>,
    w3: &RecordTensor<'a, f32, Tensor<(f32, Index), 2>, 2>,
) -> Record<'a, f32>
where
    I: TensorRef<(f32, Index), 2>,
{
    (((input * w1).map(sigmoid).unwrap() * w2).map(sigmoid).unwrap() * w3)
        .index()
        .get_as_record([0, 0])
}

/**
 * Computes mean squared loss of the network against all the training data.
 */
fn mean_squared_loss(
   inputs: &Tensor<f32, 3>,
   w1: &Tensor<f32, 2>,
   w2: &Tensor<f32, 2>,
   w3: &Tensor<f32, 2>,
   labels: &Tensor<f32, 1>,
) -> f32 {
    let inputs_shape = inputs.shape();
    let number_of_samples = inputs_shape[0].1;
    let samples_name = inputs_shape[0].0;
    {
        let mut sum = 0.0;
        for i in 0..number_of_samples {
            let input = inputs.select([(samples_name, i)]);
            let output = model(&input, w1, w2, w3);
            let correct = labels.index().get([i]);
            // sum up the squared loss
            sum = sum + ((correct - output) * (correct - output));
        }
        sum / (number_of_samples as f32)
    }
}

/**
 * Computes mean squared loss of the network against all the training data, using RecordTensor
 * types for the inputs to track derivatives.
 */
fn mean_squared_loss_training<'a>(
   inputs: &RecordTensor<'a, f32, Tensor<(f32, Index), 3>, 3>,
   w1: &RecordTensor<'a, f32, Tensor<(f32, Index), 2>, 2>,
   w2: &RecordTensor<'a, f32, Tensor<(f32, Index), 2>, 2>,
   w3: &RecordTensor<'a, f32, Tensor<(f32, Index), 2>, 2>,
   labels: &RecordTensor<'a, f32, Tensor<(f32, Index), 1>, 1>,
) -> Record<'a, f32> {
    let inputs_shape = inputs.shape();
    let number_of_samples = inputs_shape[0].1;
    let samples_name = inputs_shape[0].0;
    let history = w1.history();
    {
        let mut sum = Record::constant(0.0);
        for i in 0..number_of_samples {
            let input = inputs.view();
            let input = RecordTensor::from_existing(
                history,
                input.select([(samples_name, i)])
            );
            let output = model_training(&input, w1, w2, w3);
            let correct = labels.index().get_as_record([i]);
            // sum up the squared loss
            sum = sum + ((correct - output) * (correct - output));
        }
        sum / (number_of_samples as f32)
    }
}

/**
 * Updates the weight matrices to step the gradient by one step.
 *
 * Note that here we need the methods defined on Record / RecordTensor to do backprop. There is
 * no non-training version of this we can define without deriative tracking.
 */
fn step_gradient<'a>(
    inputs: &RecordTensor<'a, f32, Tensor<(f32, Index), 3>, 3>,
    w1: &mut RecordTensor<'a, f32, Tensor<(f32, Index), 2>, 2>,
    w2: &mut RecordTensor<'a, f32, Tensor<(f32, Index), 2>, 2>,
    w3: &mut RecordTensor<'a, f32, Tensor<(f32, Index), 2>, 2>,
    labels: &RecordTensor<'a, f32, Tensor<(f32, Index), 1>, 1>,
    learning_rate: f32,
    list: &'a WengertList<f32>
) -> f32 {
    let loss = mean_squared_loss_training(inputs, w1, w2, w3, labels);
    let derivatives = loss.derivatives();
    // update each element in the weight matrices by the derivatives
    // unwrapping here is fine because we're not changing the history of any variables
    w1.map_mut(|x| x - (derivatives[&x] * learning_rate)).unwrap();
    w2.map_mut(|x| x - (derivatives[&x] * learning_rate)).unwrap();
    w3.map_mut(|x| x - (derivatives[&x] * learning_rate)).unwrap();
    // reset gradients
    list.clear();
    w1.reset();
    w2.reset();
    w3.reset();
    // return the loss
    loss.number
}

// use a fixed seed random generator from the rand crate
let mut random_generator = rand_chacha::ChaCha8Rng::seed_from_u64(25);

// randomly initalise the weights using the fixed seed generator for reproducibility
let list = WengertList::new();
// w1 will be a 3x3 matrix
let mut w1 = RecordTensor::variables(
    &list,
    Tensor::from([("r", 3), ("c", 3)], n_random_numbers(&mut random_generator, 9))
);
// w2 will be a 3x3 matrix
let mut w2 = RecordTensor::variables(
    &list,
    Tensor::from([("r", 3), ("c", 3)], n_random_numbers(&mut random_generator, 9))
);
// w3 will be a 3x1 column matrix
// Note: We keep the shape here as 3x1 instead of just a 3 length vector to keep matrix
// multiplication simple like the Matrix example
let mut w3 = RecordTensor::variables(
    &list,
    Tensor::from([("r", 3), ("c", 1)], n_random_numbers(&mut random_generator, 3))
);
println!("w1 {}", w1);
println!("w2 {}", w2);
println!("w3 {}", w3);

// define XOR inputs, with biases added to the inputs
// again, it keeps the matrix multiplication easier if we stick to row matrices
// instead of actual vectors here
let inputs = RecordTensor::constants(
    Tensor::from([("sample", 4), ("r", 1), ("c", 3)], vec![
        0.0, 0.0, 1.0,

        0.0, 1.0, 1.0,

        1.0, 0.0, 1.0,

        1.0, 1.0, 1.0
    ])
);
// define XOR outputs which will be used as labels
let labels = RecordTensor::constants(
    Tensor::from([("sample", 4)], vec![ 0.0, 1.0, 1.0, 0.0 ])
);
let learning_rate = 0.2;
let epochs = 4000;

// do the gradient descent and save the loss at each epoch
let mut losses = Vec::with_capacity(epochs);
for _ in 0..epochs {
    losses.push(
        step_gradient(&inputs, &mut w1, &mut w2, &mut w3, &labels, learning_rate, &list)
    );
}

// now plot the training loss
let mut chart = Chart::new(180, 60, 0.0, epochs as f32);
chart.lineplot(
    &Shape::Lines(&losses.iter()
        .cloned()
        .enumerate()
        .map(|(i, x)| (i as f32, x))
        .collect::<Vec<(f32, f32)>>())
    ).display();

// note that with different hyperparameters, starting weights, or less training
// the network may not have converged and could still be outputting 0.5 for everything,
// the chart plot with this configuration is particularly interesting because the loss
// hovers around 0.3 to 0.2 for a while (while outputting 0.5 for every input) before
// finally learning how to remap the input data in a way which can then be linearly
// seperated to achieve ~0.0 loss.

// check that the weights are sensible
println!("w1 {}", w1);
println!("w2 {}", w2);
println!("w3 {}", w3);
// check that the network has learned XOR properly

{
    let inputs = inputs.view();
    let row_1 = RecordTensor::from_existing(
        Some(&list),
        inputs.select([("sample", 0)])
    );
    let row_2 = RecordTensor::from_existing(
        Some(&list),
        inputs.select([("sample", 1)])
    );
    let row_3 = RecordTensor::from_existing(
        Some(&list),
        inputs.select([("sample", 2)])
    );
    let row_4 = RecordTensor::from_existing(
        Some(&list),
        inputs.select([("sample", 3)])
    );

    println!("0 0: {:?}", model_training(&row_1, &w1, &w2, &w3).number);
    println!("0 1: {:?}", model_training(&row_2, &w1, &w2, &w3).number);
    println!("1 0: {:?}", model_training(&row_3, &w1, &w2, &w3).number);
    println!("1 1: {:?}", model_training(&row_4, &w1, &w2, &w3).number);
}
assert!(losses[epochs - 1] < 0.02);

// we can also extract the learned weights once done with training and avoid the memory
// overhead of Record
let w1_final = w1.view().map(|(x, _)| x);
let w2_final = w2.view().map(|(x, _)| x);
let w3_final = w3.view().map(|(x, _)| x);
let inputs_final = inputs.view().map(|(x, _)| x);
println!(
    "0 0: {:?}",
    model(&inputs_final.select([("sample", 0)]), &w1_final, &w2_final, &w3_final)
);
println!(
    "0 1: {:?}",
    model(&inputs_final.select([("sample", 1)]), &w1_final, &w2_final, &w3_final)
);
println!(
    "1 0: {:?}",
    model(&inputs_final.select([("sample", 2)]), &w1_final, &w2_final, &w3_final)
);
println!(
    "1 1: {:?}",
    model(&inputs_final.select([("sample", 3)]), &w1_final, &w2_final, &w3_final)
);
```

# Handwritten digit recognition on the MNIST dataset

[Web Assembly example](super::web_assembly#handwritten-digit-recognition-on-the-mnist-dataset)
 */
