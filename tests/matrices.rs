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
    fn check_empty_dimensionality() {
        let zeros = Matrix::empty(0, (4, 3));
        assert_eq!((4, 3), zeros.size());
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
    fn check_column_major_iterator() {
        let matrix = Matrix::from(vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
        println!("{:?}", matrix);
        let mut iterator = matrix.column_major_iter();
        assert_eq!(iterator.next(), Some(1));
        assert_eq!(iterator.next(), Some(2));
        assert_eq!(iterator.next(), Some(3));
        assert_eq!(iterator.next(), Some(4));
        assert_eq!(iterator.next(), Some(5));
        assert_eq!(iterator.next(), Some(6));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn check_row_major_iterator() {
        let matrix = Matrix::from(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        println!("{:?}", matrix);
        let mut iterator = matrix.row_major_iter();
        assert_eq!(iterator.next(), Some(1));
        assert_eq!(iterator.next(), Some(2));
        assert_eq!(iterator.next(), Some(3));
        assert_eq!(iterator.next(), Some(4));
        assert_eq!(iterator.next(), Some(5));
        assert_eq!(iterator.next(), Some(6));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn check_row_major_reference_iterator() {
        let matrix = Matrix::from(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        println!("{:?}", matrix);
        let mut iterator = matrix.row_major_reference_iter();
        assert_eq!(iterator.next(), Some(&1));
        assert_eq!(iterator.next(), Some(&2));
        assert_eq!(iterator.next(), Some(&3));
        assert_eq!(iterator.next(), Some(&4));
        assert_eq!(iterator.next(), Some(&5));
        assert_eq!(iterator.next(), Some(&6));
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

    #[test]
    fn check_matrix_addition() {
        let matrix1 = Matrix::from(vec![vec![-1, 2], vec![-3, 4], vec![5, -6]]);
        let matrix2 = Matrix::from(vec![vec![0, 0], vec![-3, 1], vec![3, -2]]);
        assert_eq!(matrix1 + matrix2, Matrix::from(vec![vec![-1, 2], vec![-6, 5], vec![8, -8]]));
    }

    #[test]
    #[should_panic]
    fn check_matrix_addition_wrong_size() {
        let matrix1 = Matrix::from(vec![vec![-1, 2], vec![-3, 4], vec![5, -6]]);
        let matrix2 = Matrix::from(vec![vec![0], vec![-3], vec![3]]);
        println!("{:?}", &matrix1 + &matrix2);
    }

    #[test]
    fn check_matrix_subtraction() {
        let matrix1 = Matrix::from(vec![vec![-1, 2], vec![-3, 4], vec![5, -6]]);
        let matrix2 = Matrix::from(vec![vec![0, 0], vec![-3, 1], vec![3, -2]]);
        assert_eq!(matrix1 - matrix2, Matrix::from(vec![vec![-1, 2], vec![0, 3], vec![2, -4]]));
    }

    #[test]
    fn check_matrix_negation() {
        let matrix1 = Matrix::from(vec![vec![-1, 2], vec![1, -2]]);
        assert_eq!(- matrix1, Matrix::from(vec![vec![1, -2], vec![-1, 2]]));
    }

    #[test]
    fn check_resizing_matrix() {
        let mut matrix = Matrix::from(vec![
            vec![ 1, 2 ],
            vec![ 3, 4]]);
        matrix.insert_row(0, 5);
        let mut iterator = matrix.column_major_iter();
        assert_eq!(Some(5), iterator.next());
        assert_eq!(Some(1), iterator.next());
        assert_eq!(Some(3), iterator.next());
        assert_eq!(Some(5), iterator.next());
        assert_eq!(Some(2), iterator.next());
        assert_eq!(Some(4), iterator.next());
        assert_eq!(None, iterator.next());
        matrix.remove_column(0);
        let mut iterator = matrix.column_major_iter();
        assert_eq!(Some(5), iterator.next());
        assert_eq!(Some(2), iterator.next());
        assert_eq!(Some(4), iterator.next());
        assert_eq!(None, iterator.next());
        assert_eq!((3, 1), matrix.size());
        matrix.insert_column(1, 3);
        assert_eq!((3, 2), matrix.size());
        matrix.remove_row(1);
        assert_eq!((2, 2), matrix.size());
    }
}
