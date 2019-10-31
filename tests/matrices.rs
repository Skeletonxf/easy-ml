extern crate easy_ml;

#[cfg(test)]
mod tests {
    use easy_ml::matrices::Matrix;

    #[test]
    fn check_dimensionality() {
        let row_vector = Matrix::row(vec![1, 2, 3]);
        let column_vector = Matrix::column(vec![1, 2, 3]);
        println!("{:?} {:?}", row_vector, column_vector);
        assert_eq!((1, 3), row_vector.size());
        assert_eq!((3, 1), column_vector.size());
    }

    #[test]
    fn check_dimensionality_matrix() {
        let column_vector = Matrix::from(vec![ vec![1], vec![2], vec![3] ]);
        println!("{:?}", column_vector);
        assert_eq!((3, 1), column_vector.size());
        let matrix = Matrix::from(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        println!("{:?}", matrix);
        assert_eq!((3, 2), matrix.size());
        assert_eq!((2, 3), matrix.transpose().size());
    }

    #[test]
    fn check_iterators() {
        let matrix = Matrix::from(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        println!("{:?}", matrix);
        let mut iterator = matrix.row_iter(1);
        assert_eq!(iterator.next(), Some(3));
        assert_eq!(iterator.next(), Some(4));
        assert_eq!(iterator.next(), None);
        let mut iterator = matrix.column_iter(0);
        assert_eq!(iterator.next(), Some(1));
        assert_eq!(iterator.next(), Some(3));
        assert_eq!(iterator.next(), Some(5));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn check_matrix_multiplication() {
        let matrix1 = Matrix::from(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        let matrix2 = Matrix::from(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let result = Matrix::from(vec![vec![9, 12, 15], vec![19, 26, 33], vec![29, 40, 51]]);
        assert_eq!(matrix1 * matrix2, result);
    }

    #[test]
    #[should_panic]
    fn check_matrix_multiplication_wrong_size() {
        let matrix1 = Matrix::from(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        println!("{:?}", &matrix1 * &matrix1);
    }
}
