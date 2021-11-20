extern crate easy_ml;

#[cfg(test)]
mod neural_net_tests {
    use easy_ml::differentiation::{Record, WengertList};
    use easy_ml::matrices::Matrix;
    use easy_ml::numeric::{Numeric, NumericRef};

    fn relu<T: Numeric + Copy>(x: T) -> T {
        if x > T::zero() {
            x
        } else {
            T::zero()
        }
    }

    /**
     * A simple two layer neural network that outputs a scalar.
     */
    fn model<T: Numeric + Copy>(input: &Matrix<T>, w1: &Matrix<T>, w2: &Matrix<T>) -> T
    where
        for<'a> &'a T: NumericRef<T>,
    {
        ((input * w1).map(relu) * w2).scalar()
    }

    /**
     * Computes mean squared loss of the network against the training data.
     */
    fn mean_squared_loss<T: Numeric + Copy>(
        inputs: &Vec<Matrix<T>>,
        w1: &Matrix<T>,
        w2: &Matrix<T>,
        labels: &Vec<T>,
    ) -> T
    where
        for<'a> &'a T: NumericRef<T>,
    {
        inputs
            .iter()
            .enumerate()
            .fold(T::zero(), |acc, (i, input)| {
                let output = model::<T>(input, w1, w2);
                let correct = labels[i];
                // compute and add squared loss
                acc + ((correct - output) * (correct - output))
            })
            / T::from_usize(inputs.len()).unwrap()
    }

    /**
     * Updates the weight matrices to step the gradient by one step.
     */
    fn step_gradient(
        inputs: &Vec<Matrix<Record<f32>>>,
        w1: &mut Matrix<Record<f32>>,
        w2: &mut Matrix<Record<f32>>,
        labels: &Vec<Record<f32>>,
        learning_rate: f32,
        list: &WengertList<f32>,
    ) -> f32 {
        let loss = mean_squared_loss::<Record<f32>>(inputs, w1, w2, labels);
        let derivatives = loss.derivatives();
        w1.map_mut(|x| x - (derivatives[&x] * learning_rate));
        w2.map_mut(|x| x - (derivatives[&x] * learning_rate));
        // reset gradients
        list.clear();
        w1.map_mut(Record::do_reset);
        w2.map_mut(Record::do_reset);
        loss.number
    }

    use textplots::{Chart, Plot, Shape};

    /**
     * Tests the learning of the XOR function.
     */
    #[test]
    fn test_gradient_descent() {
        let list = WengertList::new();
        let mut w1 = Matrix::from(vec![
            vec![0.1, -0.1, -0.1],
            vec![0.5, 0.4, -0.1],
            vec![0.3, 0.5, 0.2],
        ])
        .map(|x| Record::variable(x, &list));
        let mut w2 = Matrix::from(vec![vec![0.3], vec![0.1], vec![-0.4]])
            .map(|x| Record::variable(x, &list));
        // define XOR inputs, with biases added to the inputs and outputs
        let inputs = vec![
            Matrix::row(vec![0.0, 0.0, 1.0]).map(|x| Record::constant(x)),
            Matrix::row(vec![0.0, 1.0, 1.0]).map(|x| Record::constant(x)),
            Matrix::row(vec![1.0, 0.0, 1.0]).map(|x| Record::constant(x)),
            Matrix::row(vec![1.0, 1.0, 1.0]).map(|x| Record::constant(x)),
        ];
        let labels = vec![0.0, 1.0, 1.0, 0.0]
            .drain(..)
            .map(|x| Record::constant(x))
            .collect();
        let learning_rate = 0.1;
        let epochs = 300;
        let mut losses = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            losses.push(step_gradient(
                &inputs,
                &mut w1,
                &mut w2,
                &labels,
                learning_rate,
                &list,
            ))
        }
        let mut chart = Chart::new(180, 60, 0.0, epochs as f32);
        chart
            .lineplot(Shape::Lines(
                &losses
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(i, x)| (i as f32, x))
                    .collect::<Vec<(f32, f32)>>(),
            ))
            .display();
        assert!(losses[epochs - 1] < 0.02);
    }
}
