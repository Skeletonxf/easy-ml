extern crate easy_ml;

#[cfg(test)]
mod tensors {
    use easy_ml::tensors::Tensor;

    #[test]
    fn check_iterators() {
        #[rustfmt::skip]
        let tensor = Tensor::from([("row", 3), ("column", 2)], vec![
            1, 2,
            3, 4,
            5, 6
        ]);
        let mut row_column_iterator = tensor.index_order_reference_iter();
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
        let row_column = tensor.source_order();
        let mut iterator = row_column.index_order_reference_iter().with_index();
        assert_eq!(iterator.next(), Some(([0, 0], &1)));
        assert_eq!(iterator.next(), Some(([0, 1], &2)));
        assert_eq!(iterator.next(), Some(([1, 0], &3)));
        assert_eq!(iterator.next(), Some(([1, 1], &4)));
        assert_eq!(iterator.next(), Some(([2, 0], &5)));
        assert_eq!(iterator.next(), Some(([2, 1], &6)));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    #[should_panic] // FIXME: Transposition should shift the data to match the dimension order, we already have renaming/reshaping to change the shape and not the data, transposition should be the opposite
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
            Tensor::from([("b", 2), ("a", 3)], vec![
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
            tensor.transpose(["row", "column"]),
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
}
