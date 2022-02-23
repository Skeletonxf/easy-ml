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
        let mut iterator = row_column.index_reference_iter();
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
        let mut iterator = row_column.index_reference_iter().with_index();
        assert_eq!(iterator.next(), Some(([0,0], &1)));
        assert_eq!(iterator.next(), Some(([0,1], &2)));
        assert_eq!(iterator.next(), Some(([1,0], &3)));
        assert_eq!(iterator.next(), Some(([1,1], &4)));
        assert_eq!(iterator.next(), Some(([2,0], &5)));
        assert_eq!(iterator.next(), Some(([2,1], &6)));
        assert_eq!(iterator.next(), None);
    }
}
