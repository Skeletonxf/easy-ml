extern crate easy_ml;

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

        let einsum = Einsum::with_1(&x).to(["b", "a"]).unwrap();
        assert_eq!(x_transposed, also_x_transposed);
        assert_eq!(x_transposed, einsum);
    }

    #[test]
    fn summation() {
        // ab-> for X
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let sum: f32 = x.iter().sum();

        let einsum = Einsum::with_1(&x).to([]).unwrap();
        assert_eq!(sum, einsum.first());
    }

    #[test]
    fn column_sum() {
        // ab->b for X
        let x = randomish_matrix([("a", 3), ("b", 2)]);
        let column_sum = Tensor::from_fn([("b", 2)], |[b]| x.select([("b", b)]).iter().sum::<f32>());

        let einsum = Einsum::with_1(&x).to(["b"]).unwrap();
        assert_eq!(column_sum, einsum);
    }

    #[test]
    fn row_sum() {
        // ab->a for X
        let x = randomish_matrix([("a", 3), ("b", 2)]);
        let row_sum = Tensor::from_fn([("a", 3)], |[a]| x.select([("a", a)]).iter().sum::<f32>());

        let einsum = Einsum::with_1(&x).to(["a"]).unwrap();
        assert_eq!(row_sum, einsum);
    }

    #[test]
    fn matrix_vector_multiplication() {
        // ab,cb->ac for X, y
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let y = randomish_matrix([("c", 1), ("b", 3)]);
        let multiply = &x * y.transpose_view(["b", "c"]);

        let einsum = Einsum::with_2(&x, &y).to(["a", "c"]).unwrap();
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

        let einsum = Einsum::with_2(&x, &y).to(["a", "c"]).unwrap();
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

        let einsum = Einsum::with_2(&x, &x).to([]).unwrap();
        assert_eq!(dot_product, einsum.first());
    }

    #[test]
    fn elementwise_sum_with_matrices() {
        // ab,ab-> for X, X
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let sum: f32 = x.elementwise(&x, |x, y| x * y).iter().sum();

        let einsum = Einsum::with_2(&x, &x).to([]).unwrap();
        assert_eq!(sum, einsum.first());
    }

    #[test]
    fn elementwise_with_matrices() {
        // ab,ab->ab for X, X
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let multiplied = x.elementwise(&x, |x, y| x * y);

        let einsum = Einsum::with_2(&x, &x).to(["a", "b"]).unwrap();
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

        let einsum = Einsum::with_2(&x, &y).to(["a", "b"]).unwrap();
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

        let einsum = Einsum::with_2(&x, &y).to(["batch", "a", "c"]).unwrap();
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
    fn larger_output_size() {
        // ij,jk->ijk for X, Y
        let x = randomish_matrix([("i", 2), ("j", 3)]);
        let y = randomish_matrix([("j", 3), ("k", 4)]);

        let einsum = Einsum::with_2(&x, &y).to(["i", "j", "k"]).unwrap();
        assert_eq!(
            Tensor::<f32, 3>::from(
                [("i", 2), ("j", 3), ("k", 4)],
                vec![
                     0.0,   0.0,   0.0,   0.0,
                    10.0, 110.0, 210.0, 310.0,
                    40.0, 240.0, 440.0, 640.0,

                     0.0,  10.0,  20.0,  30.0,
                    11.0, 121.0, 231.0, 341.0,
                    42.0, 252.0, 462.0, 672.0,
                ],
            ),
            einsum,
        );
    }

    #[test]
    fn unrelated_dimension_inputs() {
        // ij,kl->il for X, Y
        let x = randomish_matrix([("i", 2), ("j", 3)]);
        let y = randomish_matrix([("k", 3), ("l", 4)]);

        let einsum = Einsum::with_2(&x, &y).to(["i", "l"]).unwrap();
        assert_eq!(
            Tensor::<f32, 2>::from(
                [("i", 2), ("l", 4)],
                vec![
                    90.0,  990.0, 1890.0, 2790.0,
                    99.0, 1089.0, 2079.0, 3069.0,
                ],
            ),
            einsum,
        );
    }

    #[test]
    fn three_matrix_multiplication() {
        // ab,cb,bd->ac for X, Y, Z
        let x = randomish_matrix([("a", 2), ("b", 3)]);
        let y = randomish_matrix([("c", 2), ("b", 3)]);
        let z = randomish_matrix([("b", 3), ("d", 4)]);

        let einsum = Einsum::with_3(&x, &y, &z).to(["a", "c"]).unwrap();
        assert_eq!(
            Tensor::<f32, 2>::from(
                [("a", 2), ("c", 2)],
                vec![
                    33600.0, 35600.0,
                    35600.0, 37792.0,
                ],
            ),
            einsum,
        );
    }

    #[test]
    fn four_tensor_multiplication() {
        // wx,xy,xyz,wz->xz for W, X, Y, Z
        let w = Tensor::from_fn([("w", 4), ("x", 2)], |[w,x]| ((w * 2) + x) as f32);
        let x = Tensor::from_fn([("x", 2), ("y", 3)], |[x,y]| ((x * 3) + y) as f32);
        let y = Tensor::from_fn([("x", 2), ("y", 3), ("z", 2)], |[x,y,z]| ((x * 6) + (y * 2) + z) as f32);
        let z = Tensor::from_fn([("w", 4), ("z", 2)], |[w,z]| ((w * 2) + z) as f32);

        let einsum = Einsum::with_4(&w, &x, &y, &z).to(["x", "z"]).unwrap();
        assert_eq!(
            Tensor::<f32, 2>::from(
                [("x", 2), ("z", 2)],
                vec![
                    560.0, 884.0,
                    6800.0, 9408.0,
                ],
            ),
            einsum,
        );
    }

    #[test]
    fn five_tensor_multiplication() {
        // wx,xy,xyz,wz,zyx->xz for W, X, Y, Z, Y
        let w = Tensor::from_fn([("w", 4), ("x", 2)], |[w,x]| ((w * 2) + x) as f32);
        let x = Tensor::from_fn([("x", 2), ("y", 3)], |[x,y]| ((x * 3) + y) as f32);
        let y = Tensor::from_fn([("x", 2), ("y", 3), ("z", 2)], |[x,y,z]| ((x * 6) + (y * 2) + z) as f32);
        let z = Tensor::from_fn([("w", 4), ("z", 2)], |[w,z]| ((w * 2) + z) as f32);

        let einsum = Einsum::with_5(&w, &x, &y, &z, &y)
            // NB: We can swap Y from xyz to zyx because z and x have the same
            // length. We can't do other renames like zxy because that would
            // make the lengths of the dimension names inconsistent.
            .named(["w", "x"], ["x", "y"], ["x", "y", "z"], ["w", "z"], ["z", "y", "x"])
            .to(["x", "z"])
            .unwrap();
        assert_eq!(
            Tensor::<f32, 2>::from(
                [("x", 2), ("z", 2)],
                vec![
                    2016.0, 8432.0,
                    24752.0, 90384.0,
                ],
            ),
            einsum,
        );
    }

    #[test]
    fn six_tensor_multiplication() {
        // wx,xy,xyz,wz,xyz,xy->xyz for W, X, Y, Z, Y, -X
        let w = Tensor::from_fn([("w", 4), ("x", 2)], |[w,x]| ((w * 2) + x) as f32);
        let x = Tensor::from_fn([("x", 2), ("y", 3)], |[x,y]| ((x * 3) + y) as f32);
        let y = Tensor::from_fn([("x", 2), ("y", 3), ("z", 2)], |[x,y,z]| ((x * 6) + (y * 2) + z) as f32);
        let z = Tensor::from_fn([("w", 4), ("z", 2)], |[w,z]| ((w * 2) + z) as f32);

        let einsum = Einsum::with_6(&w, &x, &y, &z, &y, &x.map(|n| -n))
            .to(["x", "y", "z"])
            .unwrap();
        assert_eq!(
            Tensor::<f32, 3>::from(
                [("x", 2), ("y", 3), ("z", 2)],
                vec![
                    0.0, 0.0,
                    -224.0, -612.0,
                    -3584.0, -6800.0,

                    -22032.0,  -37044.0,
                    -69632.0, -108864.0,
                    -170000.0, -254100.0,
                ],
            ),
            einsum,
        );
    }

    #[test]
    fn six_tensor_sum() {
        // wx,xy,xyz,wz,xyz,xy-> for W, X, Y, Z, Y, -X
        let w = Tensor::from_fn([("w", 4), ("x", 2)], |[w,x]| ((w * 2) + x) as f32);
        let x = Tensor::from_fn([("x", 2), ("y", 3)], |[x,y]| ((x * 3) + y) as f32);
        let y = Tensor::from_fn([("x", 2), ("y", 3), ("z", 2)], |[x,y,z]| ((x * 6) + (y * 2) + z) as f32);
        let z = Tensor::from_fn([("w", 4), ("z", 2)], |[w,z]| ((w * 2) + z) as f32);

        let einsum = Einsum::with_6(&w, &x, &y, &z, &y, &x.map(|n| -n))
            .to([])
            .unwrap();
        assert_eq!(Tensor::<f32, 0>::from([], vec![-672892.0]), einsum);
    }
}
