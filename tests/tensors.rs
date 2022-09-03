extern crate easy_ml;

#[cfg(test)]
mod tensors {
    use easy_ml::tensors::Tensor;

    #[test]
    fn indexing_test() {
        let tensor = Tensor::from([("x", 2), ("y", 2)], vec![1, 2, 3, 4]);
        let xy = tensor.index_by(["x", "y"]);
        let yx = tensor.index_by(["y", "x"]);
        assert_eq!(xy.get([0, 0]), 1);
        assert_eq!(xy.get([0, 1]), 2);
        assert_eq!(xy.get([1, 0]), 3);
        assert_eq!(xy.get([1, 1]), 4);
        assert_eq!(yx.get([0, 0]), 1);
        assert_eq!(yx.get([0, 1]), 3);
        assert_eq!(yx.get([1, 0]), 2);
        assert_eq!(yx.get([1, 1]), 4);
    }

    #[test]
    #[should_panic]
    fn repeated_name() {
        Tensor::from([("x", 2), ("x", 2)], vec![1, 2, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn wrong_size() {
        Tensor::from([("x", 2), ("y", 3)], vec![1, 2, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn bad_indexing() {
        let tensor = Tensor::from([("x", 2), ("y", 2)], vec![1, 2, 3, 4]);
        tensor.index_by(["x", "x"]);
    }

    #[test]
    #[rustfmt::skip]
    fn transpose_more_dimensions() {
        let tensor = Tensor::from(
            [("batch", 2), ("y", 10), ("x", 10), ("color", 1)], vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let transposed = tensor.transpose(["batch", "x", "y", "color"]);
        assert_eq!(
            transposed,
            Tensor::from([("batch", 2), ("y", 10), ("x", 10), ("color", 1)], vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
                0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ])
        );
    }

    #[test]
    fn check_iterators() {
        #[rustfmt::skip]
        let tensor = Tensor::from([("row", 3), ("column", 2)], vec![
            1, 2,
            3, 4,
            5, 6
        ]);
        let mut row_column_iterator = tensor.iter_reference();
        assert_eq!(row_column_iterator.next(), Some(&1));
        assert_eq!(row_column_iterator.next(), Some(&2));
        assert_eq!(row_column_iterator.next(), Some(&3));
        assert_eq!(row_column_iterator.next(), Some(&4));
        assert_eq!(row_column_iterator.next(), Some(&5));
        assert_eq!(row_column_iterator.next(), Some(&6));
        assert_eq!(row_column_iterator.next(), None);
    }

    #[test]
    fn check_iterators_with_index() {
        #[rustfmt::skip]
        let tensor = Tensor::from([("row", 3), ("column", 2)], vec![
            1, 2,
            3, 4,
            5, 6
        ]);
        let row_column = tensor.index();
        let mut iterator = row_column.iter_reference().with_index();
        assert_eq!(iterator.next(), Some(([0, 0], &1)));
        assert_eq!(iterator.next(), Some(([0, 1], &2)));
        assert_eq!(iterator.next(), Some(([1, 0], &3)));
        assert_eq!(iterator.next(), Some(([1, 1], &4)));
        assert_eq!(iterator.next(), Some(([2, 0], &5)));
        assert_eq!(iterator.next(), Some(([2, 1], &6)));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn check_transposition() {
        let mut tensor = Tensor::from([("row", 4), ("column", 1)], vec![1, 2, 3, 4]);
        tensor.transpose_mut(["column", "row"]);
        assert_eq!(
            tensor,
            Tensor::from([("row", 1), ("column", 4)], vec![1, 2, 3, 4])
        );
        let mut tensor = Tensor::from([("row", 1), ("column", 4)], vec![1, 2, 3, 4]);
        tensor.transpose_mut(["column", "row"]);
        assert_eq!(
            tensor,
            Tensor::from([("row", 4), ("column", 1)], vec![1, 2, 3, 4])
        );
        #[rustfmt::skip]
        let mut tensor = Tensor::from([("row", 3), ("column", 3)], vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ]);
        tensor.transpose_mut(["column", "row"]);
        #[rustfmt::skip]
        assert_eq!(
            tensor,
            Tensor::from(
                [("row", 3), ("column", 3)],
                vec![
                    1, 4, 7,
                    2, 5, 8,
                    3, 6, 9
                ]
            )
        );
        #[rustfmt::skip]
        let mut tensor = Tensor::from([("r", 2), ("c", 3)], vec![
            1, 2, 3,
            4, 5, 6
        ]);
        tensor.transpose_mut(["c", "r"]);
        #[rustfmt::skip]
        assert_eq!(
            tensor,
            Tensor::from([("r", 3), ("c", 2)], vec![
                1, 4,
                2, 5,
                3, 6
            ])
        );
        #[rustfmt::skip]
        let mut tensor = Tensor::from([("a", 3), ("b", 2)], vec![
            1, 2,
            3, 4,
            5, 6
        ]);
        tensor.transpose_mut(["b", "a"]);
        #[rustfmt::skip]
        assert_eq!(
            tensor,
            Tensor::from([("a", 2), ("b", 3)], vec![
                1, 3, 5,
                2, 4, 6
            ])
        );
        #[rustfmt::skip]
        let tensor = Tensor::from([("row", 3), ("column", 3)], vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ]);
        #[rustfmt::skip]
        assert_eq!(
            tensor.transpose(["column", "row"]),
            Tensor::from(
                [("row", 3), ("column", 3)],
                vec![
                    1, 4, 7,
                    2, 5, 8,
                    3, 6, 9
                ]
            )
        );
    }

    #[test]
    fn check_reorder() {
        let mut tensor = Tensor::from([("row", 4), ("column", 1)], vec![1, 2, 3, 4]);
        tensor.reorder_mut(["column", "row"]);
        assert_eq!(
            tensor,
            Tensor::from([("column", 1), ("row", 4)], vec![1, 2, 3, 4])
        );
        let mut tensor = Tensor::from([("row", 1), ("column", 4)], vec![1, 2, 3, 4]);
        tensor.reorder_mut(["column", "row"]);
        assert_eq!(
            tensor,
            Tensor::from([("column", 4), ("row", 1)], vec![1, 2, 3, 4])
        );
        #[rustfmt::skip]
        let mut tensor = Tensor::from([("row", 3), ("column", 3)], vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ]);
        tensor.reorder_mut(["column", "row"]);
        assert_eq!(
            tensor,
            Tensor::from(
                [("column", 3), ("row", 3)],
                vec![1, 4, 7, 2, 5, 8, 3, 6, 9,]
            )
        );
        #[rustfmt::skip]
        let mut tensor = Tensor::from([("r", 2), ("c", 3)], vec![
            1, 2, 3,
            4, 5, 6
        ]);
        tensor.reorder_mut(["c", "r"]);
        assert_eq!(
            tensor,
            Tensor::from([("c", 3), ("r", 2)], vec![1, 4, 2, 5, 3, 6,])
        );
        #[rustfmt::skip]
        let mut tensor = Tensor::from([("a", 3), ("b", 2)], vec![
            1, 2,
            3, 4,
            5, 6
        ]);
        tensor.reorder_mut(["b", "a"]);
        assert_eq!(
            tensor,
            Tensor::from([("b", 2), ("a", 3)], vec![1, 3, 5, 2, 4, 6,])
        );
        #[rustfmt::skip]
        let tensor = Tensor::from([("row", 3), ("column", 3)], vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ]);
        assert_eq!(
            tensor.reorder(["column", "row"]),
            Tensor::from(
                [("column", 3), ("row", 3)],
                vec![1, 4, 7, 2, 5, 8, 3, 6, 9,]
            )
        );
    }

    #[test]
    fn test_reshaping() {
        let tensor = Tensor::from([("everything", 20)], (0..20).collect());
        let mut five_by_four = tensor.reshape_owned([("fives", 5), ("fours", 4)]);
        #[rustfmt::skip]
        assert_eq!(
            Tensor::from([("fives", 5), ("fours", 4)], vec![
                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9, 10, 11,
                12, 13, 14, 15,
                16, 17, 18, 19
            ]),
            five_by_four
        );
        five_by_four.reshape_mut([("twos", 2), ("tens", 10)]);
        #[rustfmt::skip]
        assert_eq!(
            Tensor::from([("twos", 2), ("tens", 10)], vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19
            ]),
            five_by_four
        );
        let flattened = five_by_four.reshape_owned([("data", 20)]);
        assert_eq!(flattened, Tensor::from([("data", 20)], (0..20).collect()));
    }

    #[test]
    #[should_panic]
    fn invalid_reshape() {
        let mut square = Tensor::from([("r", 2), ("c", 2)], (0..4).collect());
        square.reshape_mut([("not", 3), ("square", 1)]);
    }

    #[test]
    fn check_data_layout_tensor() {
        use easy_ml::tensors::views::{DataLayout, TensorRef};
        let tensor = Tensor::from([("b", 3), ("r", 3), ("c", 3)], (0..27).collect());
        assert_eq!(tensor.data_layout(), DataLayout::Linear([0, 1, 2]));
        let tensor = Tensor::from([("r", 2), ("c", 2)], (0..4).collect());
        assert_eq!(tensor.data_layout(), DataLayout::Linear([0, 1]));
        let tensor = Tensor::from([("a", 3)], (0..3).collect());
        assert_eq!(tensor.data_layout(), DataLayout::Linear([0]));
    }

    #[test]
    fn check_data_layout_non_linear_tensor_views() {
        use easy_ml::tensors::views::{DataLayout, TensorRef};
        let tensor = Tensor::from([("b", 3), ("r", 3), ("c", 3)], (0..27).collect());
        assert_eq!(tensor.data_layout(), DataLayout::Linear([0, 1, 2]));
        assert_eq!(
            tensor.range([("b", 0..2)]).unwrap().source_ref().data_layout(),
            DataLayout::NonLinear
        );
        assert_eq!(
            tensor.mask([("c", 0..2)]).unwrap().source_ref().data_layout(),
            DataLayout::NonLinear
        );
        assert_eq!(
            tensor.select([("b", 1)]).source_ref().data_layout(),
            DataLayout::NonLinear
        );
        assert_eq!(
            tensor.expand([(2, "x")]).source_ref().data_layout(),
            DataLayout::NonLinear
        );
    }

    #[test]
    #[should_panic] // FIXME: TensorAccess is implemented wrong here
    fn check_data_layout_tensor_access() {
        use easy_ml::tensors::views::{DataLayout, TensorRef};
        let tensor = Tensor::from([("b", 3), ("r", 3), ("c", 3)], (0..27).collect());
        assert_eq!(tensor.data_layout(), DataLayout::Linear([0, 1, 2]));
        assert_eq!(
            tensor.index_by(["b", "r", "c"]).data_layout(),
            DataLayout::Linear([0, 1, 2])
        );
        assert_eq!(
            tensor.index_by(["c", "r", "b"]).data_layout(),
            DataLayout::Linear([2, 1, 0])
        );
        assert_eq!( // b = 0, r == 1, c == 2, => 1, 2, 0
            tensor.index_by(["r", "c", "b"]).data_layout(),
            DataLayout::Linear([1, 2, 0])
        );
        assert_eq!(
            tensor.index_by(["c", "r", "b"]).data_layout(),
            DataLayout::Linear([2, 1, 0])
        );
        assert_eq!(
            tensor.index_by(["r", "b", "c"]).data_layout(),
            DataLayout::Linear([1, 0, 2])
        );
        assert_eq!(
            tensor.index_by(["c", "b", "r"]).data_layout(),
            DataLayout::Linear([2, 0, 1])
        );
        assert_eq!(
            tensor.transpose_view(["b", "r", "c"]).source_ref().data_layout(),
            DataLayout::Linear([0, 1, 2])
        );
        assert_eq!(
            tensor.transpose_view(["c", "r", "b"]).source_ref().data_layout(),
            DataLayout::Linear([2, 1, 0])
        );
        assert_eq!(
            tensor.transpose_view(["r", "c", "b"]).source_ref().data_layout(),
            DataLayout::Linear([1, 2, 0])
        );
        assert_eq!(
            tensor.transpose_view(["c", "r", "b"]).source_ref().data_layout(),
            DataLayout::Linear([2, 1, 0])
        );
        assert_eq!(
            tensor.transpose_view(["r", "b", "c"]).source_ref().data_layout(),
            DataLayout::Linear([1, 0, 2])
        );
        assert_eq!(
            tensor.transpose_view(["c", "b", "r"]).source_ref().data_layout(),
            DataLayout::Linear([2, 0, 1])
        );
    }

    #[test]
    fn check_data_layout_linear_tensor_views() {
        use easy_ml::tensors::views::{DataLayout, TensorRef};
        let tensor = Tensor::from([("b", 3), ("r", 3), ("c", 3)], (0..27).collect());
        assert_eq!(tensor.data_layout(), DataLayout::Linear([0, 1, 2]));
        assert_eq!(
            tensor.rename_view(["a", "q", "b"]).source_ref().data_layout(),
            DataLayout::Linear([0, 1, 2])
        );
    }
}
