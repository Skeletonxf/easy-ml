extern crate easy_ml;

/**
 * Temporarily allowing unused variables to get test cases together for
 * einsum tests, first just setting up with doing the regular way.
 */
#[allow(unused_variables)]
#[cfg(test)]
mod einsum {
    use easy_ml::tensors::{Tensor, Dimension};
    use easy_ml::tensors::views::TensorView;
    use easy_ml::tensors::einsum::Einsum;

    fn randomish_matrix(shape: [(Dimension, usize); 2]) -> Tensor<f32, 2> {
        Tensor::from_fn(shape, |[x, y]| (x + (10 * y)) as f32)
    }

    #[test]
    fn transpose() {
        // ab->ba for X
        let x = randomish_matrix([("a", 3), ("b", 2)]);
        // Easy ML's transpose method does not swap the dimension names along with
        // the data, but we can verify we're seeing what's expected by checking
        // the same operation with TensorAccess, which swaps the dimension names
        // in addition.
        let x_transposed = TensorView::from(x.index_by(["b", "a"]));
        let also_x_transposed = x.transpose(["b", "a"]).rename_owned(["b", "a"]);

        let einsum = Einsum::naive().with_1(&x).to(["b", "a"]).unwrap();
        assert_eq!(x_transposed, also_x_transposed);
        assert_eq!(x_transposed, einsum);
    }

    #[test]
    fn summation() {
        // ab-> for X
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let sum: f32 = x.iter().sum();

        let einsum = Einsum::naive().with_1(&x).to([]).unwrap();
        assert_eq!(sum, einsum.first());
    }

    #[test]
    fn column_sum() {
        // ab->b for X
        let x = randomish_matrix([("a", 3), ("b", 2)]);
        let column_sum = Tensor::from_fn([("b", 2)], |[b]| x.select([("b", b)]).iter().sum::<f32>());

        let einsum = Einsum::naive().with_1(&x).to(["b"]).unwrap();
        assert_eq!(column_sum, einsum);
    }

    #[test]
    fn row_sum() {
        // ab->a for X
        let x = randomish_matrix([("a", 3), ("b", 2)]);
        let row_sum = Tensor::from_fn([("a", 3)], |[a]| x.select([("a", a)]).iter().sum::<f32>());

        let einsum = Einsum::naive().with_1(&x).to(["a"]).unwrap();
        assert_eq!(row_sum, einsum);
    }

    #[test]
    fn matrix_vector_multiplication() {
        // ab,cb->ac for X, y
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let y = randomish_matrix([("c", 1), ("b", 3)]);
        let multiply = &x * y.transpose_view(["b", "c"]);

        let einsum = Einsum::naive().with_2(&x, &y).to(["a", "c"]).unwrap();
        // The transpose methods in Easy ML deliberately don't swap dimension
        // names too because a core use case is transposing the data in a matrix
        // that was already named correctly to do a multiplication with. We can verify
        // the calculation is correct if we rename the output of the einsum to the
        // ["a", "b"] dimensions calculated by the multiply.
        assert_eq!(multiply, einsum.rename_view(["a", "b"]));
    }

    #[test]
    fn matrix_multiplication() {
        // ab,cb->ac for X, Y
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let y = randomish_matrix([("c", 2), ("b", 3)]);
        let multiply = &x * y.transpose_view(["b", "c"]);

        let einsum = Einsum::naive().with_2(&x, &y).to(["a", "c"]).unwrap();
        // Again, to make the multiply operation defined we had to swap the
        // data to make y have a shape of [("c", 3), ("b", 2)] which einsum
        // completely skips, so the output has the same data in both cases
        // but different dimension names.
        assert_eq!(multiply, einsum.rename_view(["a", "b"]));
    }

    #[test]
    fn dot_product() {
        // a,a-> for x, x
        let x = Tensor::from_fn([("a", 3)], |[a]| a as f32);
        let dot_product = x.scalar_product(&x);

        let einsum = Einsum::naive().with_2(&x, &x).to([]).unwrap();
        assert_eq!(dot_product, einsum.first());
    }

    #[test]
    fn elementwise_sum_with_matrices() {
        // ab,ab-> for X, X
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let sum: f32 = x.elementwise(&x, |x, y| x * y).iter().sum();

        let einsum = Einsum::naive().with_2(&x, &x).to([]).unwrap();
        assert_eq!(sum, einsum.first());
    }

    #[test]
    fn elementwise_with_matrices() {
        // ab,ab->ab for X, X
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let multiplied = x.elementwise(&x, |x, y| x * y);

        let einsum = Einsum::naive().with_2(&x, &x).to(["a", "b"]).unwrap();
        assert_eq!(multiplied, einsum);
    }

    #[test]
    fn outer_product() {
        // a,b->ab for x, y
        let x = Tensor::from_fn([("a", 3)], |[a]| a as f32);
        let y = Tensor::from_fn([("b", 5)], |[b]| (b * 10) as f32);
        let x_indexing = x.index();
        let y_indexing = y.index();
        let dot_product = Tensor::from_fn([("a", 3), ("b", 5)], |[a, b]| x_indexing.get([a]) * y_indexing.get([b]));

        let einsum = Einsum::naive().with_2(&x, &y).to(["a", "b"]).unwrap();
        assert_eq!(dot_product, einsum);
    }

    #[test]
    fn batch_matrix_multiplication() {
        use easy_ml::tensors::views::{TensorView, TensorStack};

        // batch a b,batch b c->batch a c for X, Y
        let x = Tensor::from_fn([("batch", 3), ("a", 2), ("b", 5)], |[i,j,k]| (i + (10 * j) + (2 * k)) as f32);
        let y = Tensor::from_fn([("batch", 3), ("b", 5), ("c", 3)], |[i,j,k]| (i + (10 * j) + (2 * k)) as f32);
        let batch = {
            let mut matrices = Vec::with_capacity(3);
            for batch in 0..3 {
                let multiplied = x.select([("batch", batch)]) * y.select([("batch", batch)]);
                matrices.push(multiplied);
            }
            matrices
        };

        let einsum = Einsum::naive().with_2(&x, &y).to(["batch", "a", "c"]).unwrap();
        assert_eq!(batch[0], einsum.select([("batch", 0)]));
        assert_eq!(batch[1], einsum.select([("batch", 1)]));
        assert_eq!(batch[2], einsum.select([("batch", 2)]));
        assert_eq!(3, einsum.shape()[0].1);
        assert_eq!(
            TensorView::from(
                TensorStack::<_, (_, _, _), 2>::from(
                    (&batch[0], &batch[1], &batch[2]),
                    (0, "batch")
                )
            ),
            einsum,
        );
    }

    #[test]
    fn diagonal() {
        // aa->a for X
        let x = randomish_matrix([("a", 3), ("b", 3)]);
        let diagonal = Tensor::from([("a", 3)], x.into_matrix().diagonal_iter().map(|x| x * x).collect());

        let x = randomish_matrix([("a", 3), ("b", 3)]);
        // TODO: Need to bypass validation logic in TensorRename before we can try this
        // let einsum = Einsum::naive().with_1(&x).named(["a", "a"]).to(["a"]).unwrap();
        // assert_eq!(diagonal, einsum);
    }

    #[test]
    fn sum_of_diagonal() {
        // aa-> for X
        let x = randomish_matrix([("a", 3), ("b", 3)]);
        let sum: f32 = x.into_matrix().diagonal_iter().map(|x| x * x).sum();

        let x = randomish_matrix([("a", 3), ("b", 3)]);
        // TODO: Need to bypass validation logic in TensorRename before we can try this
        // let einsum = Einsum::naive().with_1(&x).named(["a", "a"]).to([]).unwrap();
        // assert_eq!(sum, einsum.first());
    }

    // Try 'ij,jk->ijk' for two matrices too?
    // Try a test with unrelated indexes, something like ij,kl->il
}
