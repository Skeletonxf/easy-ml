extern crate easy_ml;

#[cfg(test)]
mod tests {
    use easy_ml::matrices::Matrix;
    use easy_ml::matrices::slices::Slice;

    #[test]
    fn test_slicing_row() {
        let matrix = Matrix::from(vec![
            vec![ 1, 2],
            vec![ 3, 4]]);
        assert_eq!(matrix.retain(Slice::SingleRow(1)), Matrix::row(vec![ 3, 4 ]));
    }

    #[test]
    fn test_slicing_column() {
        let matrix = Matrix::from(vec![
            vec![ 1, 2],
            vec![ 3, 4]]);
        assert_eq!(matrix.retain(Slice::SingleColumn(0)), Matrix::column(vec![ 1, 3 ]));
    }

    #[test]
    fn test_slicing_column_range() {
        let matrix = Matrix::from(vec![
            vec![ 1, 2, 3],
            vec![ 4, 5, 6]]);
        assert_eq!(
            matrix.retain(Slice::ColumnRange(1..3)),
            Matrix::from(vec![
                vec![ 2, 3],
                vec![ 5, 6 ]]));
    }

    #[test]
    fn test_slicing_row_range() {
        let matrix = Matrix::column(vec![ 1, 2, 3, 4, 5, 6]);
        assert_eq!(
            matrix.retain(Slice::RowRange(2..8)),
            Matrix::column(vec![ 3, 4, 5, 6 ]));
    }

    #[test]
    fn test_slicing_row_column() {
        let matrix = Matrix::from(vec![
            vec![ 1, 2, 3, 4, 5, 6 ],
            vec![ 7, 8, 9, 1, 2, 3 ]]);
        assert_eq!(
            matrix.retain(Slice::RowColumnRange(0..2, 3..5)),
            Matrix::from(vec![
                vec![ 4, 5 ],
                vec![ 1, 2 ]]));
    }

    #[test]
    fn test_slicing_row_column_negated() {
        let matrix = Matrix::from(vec![
            vec![ 1, 2, 3, 4, 5, 6 ],
            vec![ 7, 8, 9, 1, 2, 3 ]]);
        assert_eq!(
            matrix.retain(Slice::RowColumnRange(1..2, 2..3).not()),
            Matrix::row(vec![ 1, 2, 4, 5, 6 ]));
    }

    #[test]
    fn test_slicing_row_column_or() {
        let matrix = Matrix::from(vec![
            vec![ 1, 2, 3 ],
            vec![ 4, 5, 6 ],
            vec![ 7, 8, 9 ]]);
        assert_eq!(
            matrix.retain(Slice::RowRange(0..1).or(Slice::RowColumnRange(2..3, 0..1))),
            Matrix::from(vec![
                vec![ 1, 2, 3],
                vec![ 7, 8, 9]]));
    }

    #[test]
    fn test_slicing_row_column_and() {
        let matrix = Matrix::from(vec![
            vec![ 1, 2, 3 ],
            vec![ 4, 5, 6 ],
            vec![ 7, 8, 9 ]]);
        assert_eq!(
            matrix.retain(Slice::RowRange(0..1).and(Slice::ColumnRange(0..2))),
            Matrix::row(vec![ 1, 2 ]));
    }
}
