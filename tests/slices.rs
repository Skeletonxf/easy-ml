extern crate easy_ml;

#[cfg(test)]
mod tests {
    use easy_ml::matrices::slices::{Slice, Slice2D};
    use easy_ml::matrices::Matrix;

    #[test]
    fn test_slicing_row() {
        #[rustfmt::skip]
        let matrix = Matrix::from(vec![
            vec![ 1, 2],
            vec![ 3, 4]
        ]);
        assert_eq!(
            matrix.retain(Slice2D::new().rows(Slice::Single(1)).columns(Slice::All())),
            Matrix::row(vec![3, 4])
        );
    }

    #[test]
    fn test_slicing_column() {
        #[rustfmt::skip]
        let matrix = Matrix::from(vec![
            vec![ 1, 2],
            vec![ 3, 4]
        ]);
        assert_eq!(
            matrix.retain(Slice2D::new().rows(Slice::All()).columns(Slice::Single(1))),
            Matrix::column(vec![2, 4])
        );
    }

    #[test]
    fn test_slicing_column_range() {
        #[rustfmt::skip]
        let matrix = Matrix::from(vec![
            vec![ 1, 2, 3],
            vec![ 4, 5, 6]
        ]);
        assert_eq!(
            matrix.retain(
                Slice2D::new()
                    .rows(Slice::All())
                    .columns(Slice::Range(1..3))
            ),
            Matrix::from(vec![vec![2, 3], vec![5, 6]])
        );
    }

    #[test]
    fn test_slicing_row_range() {
        let matrix = Matrix::column(vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(
            matrix.retain(
                Slice2D::new()
                    .rows(Slice::Range(2..8))
                    .columns(Slice::All())
            ),
            Matrix::column(vec![3, 4, 5, 6])
        );
    }

    #[test]
    fn test_slicing_row_column() {
        #[rustfmt::skip]
        let matrix = Matrix::from(vec![
            vec![ 1, 2, 3, 4, 5, 6 ],
            vec![ 7, 8, 9, 1, 2, 3 ]
        ]);
        assert_eq!(
            matrix.retain(
                Slice2D::new()
                    .rows(Slice::Range(0..2))
                    .columns(Slice::Range(3..5))
            ),
            Matrix::from(vec![vec![4, 5], vec![1, 2]])
        );
    }

    #[test]
    fn test_slicing_row_column_negated() {
        #[rustfmt::skip]
        let matrix = Matrix::from(vec![
            vec![ 1, 2, 3, 4, 5, 6 ],
            vec![ 7, 8, 9, 1, 2, 3 ]
        ]);
        assert_eq!(
            matrix.retain(
                Slice2D::new()
                    .rows(Slice::Range(1..2).not())
                    .columns(Slice::Range(2..3).not())
            ),
            Matrix::row(vec![1, 2, 4, 5, 6])
        );
    }

    #[test]
    fn test_slicing_row_column_or() {
        #[rustfmt::skip]
        let matrix = Matrix::from(vec![
            vec![ 1, 2, 3 ],
            vec![ 4, 5, 6 ],
            vec![ 7, 8, 9 ]
        ]);
        assert_eq!(
            matrix.retain(
                Slice2D::new()
                    .rows(Slice::Range(0..1).or(Slice::Range(1..2)))
                    .columns(Slice::Range(2..3))
            ),
            Matrix::from(vec![vec![3], vec![6]])
        );
    }

    #[test]
    #[should_panic]
    fn test_slicing_empty_slice_construction() {
        #[rustfmt::skip]
        let matrix = Matrix::from(vec![
            vec![ 1, 2, 3 ],
            vec![ 4, 5, 6 ],
            vec![ 7, 8, 9 ]
        ]);
        matrix.retain(
            Slice2D::new()
                .rows(Slice::Range(0..1).and(Slice::Range(1..2)))
                .columns(Slice::All()),
        );
    }
}
