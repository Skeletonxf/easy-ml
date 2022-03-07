extern crate easy_ml;

#[cfg(test)]
mod tensors {
    use easy_ml::tensors::Tensor;

    #[test]
    fn check_iterators() {
        let tensor = Tensor::from([("row", 3), ("column", 2)], vec![
            1, 2,
            3, 4,
            5, 6
        ]);
        let row_column = tensor.source_order();
        let mut iterator = row_column.index_order_reference_iter();
        assert_eq!(iterator.next(), Some(&1));
        assert_eq!(iterator.next(), Some(&2));
        assert_eq!(iterator.next(), Some(&3));
        assert_eq!(iterator.next(), Some(&4));
        assert_eq!(iterator.next(), Some(&5));
        assert_eq!(iterator.next(), Some(&6));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn check_iterators_with_index() {
        let tensor = Tensor::from([("row", 3), ("column", 2)], vec![
            1, 2,
            3, 4,
            5, 6
        ]);
        let row_column = tensor.source_order();
        let mut iterator = row_column.index_order_reference_iter().with_index();
        assert_eq!(iterator.next(), Some(([0,0], &1)));
        assert_eq!(iterator.next(), Some(([0,1], &2)));
        assert_eq!(iterator.next(), Some(([1,0], &3)));
        assert_eq!(iterator.next(), Some(([1,1], &4)));
        assert_eq!(iterator.next(), Some(([2,0], &5)));
        assert_eq!(iterator.next(), Some(([2,1], &6)));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn check_transposition() {
        let mut tensor = Tensor::from([("row", 4), ("column", 1)], vec![1, 2, 3, 4]);
        tensor.transpose_mut(["column", "row"]);
        assert_eq!(tensor, Tensor::from([("column", 1), ("row", 4)], vec![1, 2, 3, 4]));
        let mut tensor = Tensor::from([("row", 1), ("column", 4)], vec![1, 2, 3, 4]);
        tensor.transpose_mut(["column", "row"]);
        assert_eq!(tensor, Tensor::from([("column", 4), ("row", 1)], vec![1, 2, 3, 4]));
        let mut tensor = Tensor::from([("row", 3), ("column", 3)], vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ]);
        tensor.transpose_mut(["column", "row"]);
        assert_eq!(tensor, Tensor::from([("column", 3), ("row", 3)], vec![
            1, 4, 7,
            2, 5, 8,
            3, 6, 9,
        ]));
        let mut tensor = Tensor::from([("r", 2), ("c", 3)], vec![
            1, 2, 3,
            4, 5, 6,
        ]);
        tensor.transpose_mut(["c", "r"]);
        assert_eq!(tensor, Tensor::from([("c", 3), ("r", 2)], vec![
            1, 4,
            2, 5,
            3, 6,
        ]));
        let mut tensor = Tensor::from([("a", 3), ("b", 2)], vec![
            1, 2,
            3, 4,
            5, 6,
        ]);
        tensor.transpose_mut(["b", "a"]);
        assert_eq!(tensor, Tensor::from([("b", 2), ("a", 3)], vec![
            1, 3, 5,
            2, 4, 6,
        ]));
        let tensor = Tensor::from([("row", 3), ("column", 3)], vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ]);
        assert_eq!(
            tensor.transpose(["column", "row"]),
            Tensor::from([("column", 3), ("row", 3)], vec![
                1, 4, 7,
                2, 5, 8,
                3, 6, 9,
            ])
        );
    }
}
