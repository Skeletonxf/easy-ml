extern crate easy_ml;

#[cfg(test)]
mod tensors {
    use easy_ml::tensors::indexing::ShapeIterator;

    #[test]
    fn test_shape_iterator_exact_size() {
        let mut iterator = ShapeIterator::from([("x", 3), ("y", 2)]);
        assert_eq!(iterator.size_hint(), (6, Some(6)));

        let a = iterator.next();
        assert_eq!(a, Some([0, 0]));
        assert_eq!(iterator.size_hint(), (5, Some(5)));

        let b = iterator.next();
        assert_eq!(b, Some([0, 1]));
        assert_eq!(iterator.size_hint(), (4, Some(4)));

        let c = iterator.next();
        assert_eq!(c, Some([1, 0]));
        assert_eq!(iterator.size_hint(), (3, Some(3)));

        let d = iterator.next();
        assert_eq!(d, Some([1, 1]));
        assert_eq!(iterator.size_hint(), (2, Some(2)));

        let e = iterator.next();
        assert_eq!(e, Some([2, 0]));
        assert_eq!(iterator.size_hint(), (1, Some(1)));

        let f = iterator.next();
        assert_eq!(f, Some([2, 1]));
        assert_eq!(iterator.size_hint(), (0, Some(0)));

        let g = iterator.next();
        assert_eq!(g, None);
        assert_eq!(iterator.size_hint(), (0, Some(0)));
    }

    #[test]
    fn higher_dimensional_shape_iterator_len_test() {
        use std::iter::ExactSizeIterator;

        let mut iterator = ShapeIterator::from(
            [("a", 1), ("b", 2), ("c", 1), ("d", 2), ("e", 1), ("f", 2)]
        );
        assert_eq!(iterator.len(), 8);

        let a = iterator.next();
        assert_eq!(a, Some([0, 0, 0, 0, 0, 0]));
        assert_eq!(iterator.len(), 7);

        let b = iterator.next();
        assert_eq!(b, Some([0, 0, 0, 0, 0, 1]));
        assert_eq!(iterator.len(), 6);

        let c = iterator.next();
        assert_eq!(c, Some([0, 0, 0, 1, 0, 0]));
        assert_eq!(iterator.len(), 5);

        let d = iterator.next();
        assert_eq!(d, Some([0, 0, 0, 1, 0, 1]));
        assert_eq!(iterator.len(), 4);

        let e = iterator.next();
        assert_eq!(e, Some([0, 1, 0, 0, 0, 0]));
        assert_eq!(iterator.len(), 3);

        let f = iterator.next();
        assert_eq!(f, Some([0, 1, 0, 0, 0, 1]));
        assert_eq!(iterator.len(), 2);

        let g = iterator.next();
        assert_eq!(g, Some([0, 1, 0, 1, 0, 0]));
        assert_eq!(iterator.len(), 1);

        let h = iterator.next();
        assert_eq!(h, Some([0, 1, 0, 1, 0, 1]));
        assert_eq!(iterator.len(), 0);

        let i = iterator.next();
        assert_eq!(i, None);
        assert_eq!(iterator.len(), 0);
    }
}
