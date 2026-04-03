extern crate easy_ml;

/**
 * Temporarily allowing unused variables to get test cases together for
 * einsum tests, first just setting up with doing the regular way.
 */
#[allow(unused_variables)]
#[cfg(test)]
mod einsum {
    use easy_ml::tensors::{Tensor, Dimension};

    fn randomish_matrix(shape: [(Dimension, usize); 2]) -> Tensor<f32, 2> {
        Tensor::from_fn(shape, |[x, y]| (x + (10 * y)) as f32)
    }

    #[test]
    fn transpose() {
        // ab->ba for X
        let x = randomish_matrix([("a", 3), ("b", 2)]);
        let x_transposed = x.transpose(["b", "a"]);
    }

    #[test]
    fn summation() {
        // ab-> for X
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let sum: f32 = x.iter().sum();
    }

    #[test]
    fn column_sum() {
        // ab->b for X
        let x = randomish_matrix([("a", 3), ("b", 2)]);
        let column_sum = Tensor::from_fn([("b", 2)], |[b]| x.select([("b", b)]).iter().sum::<f32>());
    }

    #[test]
    fn row_sum() {
        // ab->a for X
        let x = randomish_matrix([("a", 3), ("b", 2)]);
        let row_sum = Tensor::from_fn([("a", 3)], |[a]| x.select([("a", a)]).iter().sum::<f32>());
    }

    #[test]
    fn matrix_vector_multiplication() {
        // ab,cb->ac for X, y
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let y = randomish_matrix([("c", 1), ("b", 3)]);
        let multiply = x * y.transpose_view(["b", "c"]);
    }

    #[test]
    fn matrix_multiplication() {
        // ab,cb->ac for X, Y
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let y = randomish_matrix([("c", 2), ("b", 3)]);
        let multiply = x * y.transpose_view(["b", "c"]);
        // Doing einstein summation notation for some cases like this might run into
        // the notation wanting unique dimension names for the 'same' one, but there
        // is already lots of existing support under the APIs like rename to
        // preprocess the tensor before we would use it in the einsum notation
        // so it doesn't seem worth complicating a future einsum API over.
    }

    #[test]
    fn dot_product() {
        // a,a-> for x, x
        let x = Tensor::from_fn([("a", 3)], |[a]| a as f32);
        let dot_product = x.scalar_product(&x);
    }

    #[test]
    fn elementwise_sum_with_matrices() {
        // ab,ab-> for X, X
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let sum: f32 = x.elementwise(&x, |x, y| x * y).iter().sum();
    }

    #[test]
    fn elementwise_with_matrices() {
        // ab,ab->ab for X, X
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let multiplied = x.elementwise(&x, |x, y| x * y);
    }

    #[test]
    fn outer_product() {
        // a,b->ab for x, y
        let x = Tensor::from_fn([("a", 3)], |[a]| a as f32);
        let y = Tensor::from_fn([("b", 5)], |[b]| (b * 10) as f32);
        let x_indexing = x.index();
        let y_indexing = y.index();
        let dot_product = Tensor::from_fn([("a", 3), ("b", 5)], |[a, b]| x_indexing.get([a]) * y_indexing.get([b]));
    }

    #[test]
    fn batch_matrix_multiplication() {
        // batch a b,batch b c->batch a c for X, Y
        let x = Tensor::from_fn([("batch", 3), ("a", 2), ("b", 5)], |[i,j,k]| (i + (10 * j) + (2 * k)) as f32);
        let y = Tensor::from_fn([("batch", 3), ("b", 5), ("c", 3)], |[i,j,k]| (i + (10 * j) + (2 * k)) as f32);
        let batch = {
            // TODO: Need to actually put this back together
            for batch in 0..3 {
                let multiplied = x.select([("batch", batch)]) * y.select([("batch", batch)]);
            }
        };
    }

    #[test]
    fn diagonal() {
        // aa->a for X
        let x = randomish_matrix([("a", 3), ("b", 3)]);
        let diagonal = Tensor::from([("a", 3)], x.into_matrix().diagonal_iter().map(|x| x * x).collect());
    }

    #[test]
    fn sum_of_diagonal() {
        // aa-> for X
        let x = randomish_matrix([("a", 3), ("b", 3)]);
        let sum: f32 = x.into_matrix().diagonal_iter().map(|x| x * x).sum();
    }

    // Try 'ij,jk->ijk' for two matrices too?
}
