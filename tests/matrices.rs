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
        assert_eq!(iterator.size_hint(), (2, Some(2)));
        assert_eq!(iterator.next(), Some(3));
        assert_eq!(iterator.size_hint(), (1, Some(1)));
        assert_eq!(iterator.next(), Some(4));
        assert_eq!(iterator.size_hint(), (0, Some(0)));
        assert_eq!(iterator.next(), None);
        assert_eq!(iterator.size_hint(), (0, Some(0)));
        let mut iterator = matrix.column_iter(0);
        assert_eq!(iterator.size_hint(), (3, Some(3)));
        assert_eq!(iterator.next(), Some(1));
        assert_eq!(iterator.size_hint(), (2, Some(2)));
        assert_eq!(iterator.next(), Some(3));
        assert_eq!(iterator.size_hint(), (1, Some(1)));
        assert_eq!(iterator.next(), Some(5));
        assert_eq!(iterator.size_hint(), (0, Some(0)));
        assert_eq!(iterator.next(), None);
        assert_eq!(iterator.size_hint(), (0, Some(0)));
    }

    #[test]
    fn check_column_major_iterator() {
        let matrix = Matrix::from(vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
        println!("{:?}", matrix);
        let mut iterator = matrix.column_major_iter();
        assert_eq!(iterator.size_hint(), (6, Some(6)));
        assert_eq!(iterator.next(), Some(1));
        assert_eq!(iterator.size_hint(), (5, Some(5)));
        assert_eq!(iterator.next(), Some(2));
        assert_eq!(iterator.size_hint(), (4, Some(4)));
        assert_eq!(iterator.next(), Some(3));
        assert_eq!(iterator.size_hint(), (3, Some(3)));
        assert_eq!(iterator.next(), Some(4));
        assert_eq!(iterator.size_hint(), (2, Some(2)));
        assert_eq!(iterator.next(), Some(5));
        assert_eq!(iterator.size_hint(), (1, Some(1)));
        assert_eq!(iterator.next(), Some(6));
        assert_eq!(iterator.size_hint(), (0, Some(0)));
        assert_eq!(iterator.next(), None);
        assert_eq!(iterator.size_hint(), (0, Some(0)));
    }

    #[test]
    fn check_row_major_iterator() {
        let matrix = Matrix::from(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        println!("{:?}", matrix);
        let mut iterator = matrix.row_major_iter();
        assert_eq!(iterator.size_hint(), (6, Some(6)));
        assert_eq!(iterator.next(), Some(1));
        assert_eq!(iterator.size_hint(), (5, Some(5)));
        assert_eq!(iterator.next(), Some(2));
        assert_eq!(iterator.size_hint(), (4, Some(4)));
        assert_eq!(iterator.next(), Some(3));
        assert_eq!(iterator.size_hint(), (3, Some(3)));
        assert_eq!(iterator.next(), Some(4));
        assert_eq!(iterator.size_hint(), (2, Some(2)));
        assert_eq!(iterator.next(), Some(5));
        assert_eq!(iterator.size_hint(), (1, Some(1)));
        assert_eq!(iterator.next(), Some(6));
        assert_eq!(iterator.size_hint(), (0, Some(0)));
        assert_eq!(iterator.next(), None);
        assert_eq!(iterator.size_hint(), (0, Some(0)));
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
        let mut iterator = matrix.row_major_iter();
        assert_eq!(Some(5), iterator.next());
        assert_eq!(Some(3), iterator.next());
        assert_eq!(Some(2), iterator.next());
        assert_eq!(Some(3), iterator.next());
        assert_eq!(Some(4), iterator.next());
        assert_eq!(Some(3), iterator.next());
        assert_eq!(None, iterator.next());
        assert_eq!((3, 2), matrix.size());
        matrix.remove_row(1);
        let mut iterator = matrix.row_major_iter();
        assert_eq!(Some(5), iterator.next());
        assert_eq!(Some(3), iterator.next());
        assert_eq!(Some(4), iterator.next());
        assert_eq!(Some(3), iterator.next());
        assert_eq!(None, iterator.next());
        assert_eq!((2, 2), matrix.size());
    }

    #[test]
    fn check_growing_matrix() {
        let mut matrix = Matrix::from_scalar(5);
        matrix.insert_row(1, 3);
        assert_eq!(matrix, Matrix::from(vec![
            vec![ 5 ],
            vec![ 3 ]
        ]));
        matrix.insert_column(0, 4);
        assert_eq!(matrix, Matrix::from(vec![
            vec![ 4, 5 ],
            vec![ 4, 3 ]
        ]));
        matrix.insert_row_with(0, [1, 2].iter().cloned());
        assert_eq!(matrix, Matrix::from(vec![
            vec![ 1, 2 ],
            vec![ 4, 5 ],
            vec![ 4, 3 ]
        ]));
        matrix.insert_row_with(2, [7, 8, 9].iter().cloned());
        assert_eq!(matrix, Matrix::from(vec![
            vec![ 1, 2 ],
            vec![ 4, 5 ],
            vec![ 7, 8 ],
            vec![ 4, 3 ]
        ]));
        matrix.insert_column_with(2, [6, 0, 3, 7].iter().cloned());
        assert_eq!(matrix, Matrix::from(vec![
            vec![ 1, 2, 6 ],
            vec![ 4, 5, 0 ],
            vec![ 7, 8, 3 ],
            vec![ 4, 3, 7 ]
        ]));
    }

    #[test]
    #[should_panic]
    fn check_insert_column_with_too_few_elements() {
        let mut matrix = Matrix::column(vec![ 1, 2, 3 ]);
        matrix.insert_column_with(1, [4, 5].iter().cloned());
    }

    #[test]
    #[should_panic]
    fn check_insert_row_with_too_few_elements() {
        let mut matrix = Matrix::row(vec![ 1, 2, 3 ]);
        matrix.insert_row_with(0, [4, 5].iter().cloned());
    }

    #[test]
    fn check_shrinking_matrix() {
        let mut matrix = Matrix::from(vec![
            vec![ 1, 2, 6 ],
            vec![ 4, 5, 0 ],
            vec![ 7, 8, 3 ],
            vec![ 4, 3, 7 ]
        ]);
        matrix.remove_column(0);
        assert_eq!(matrix, Matrix::from(vec![
            vec![ 2, 6 ],
            vec![ 5, 0 ],
            vec![ 8, 3 ],
            vec![ 3, 7 ]
        ]));
        matrix.remove_row(1);
        assert_eq!(matrix, Matrix::from(vec![
            vec![ 2, 6 ],
            vec![ 8, 3 ],
            vec![ 3, 7 ]
        ]));
    }

    #[test]
    fn check_mapping() {
        let mut matrix = Matrix::from(vec![
            vec![ 0, 1, 2 ],
            vec![ 3, 4, 5 ],
            vec![ 6, 7, 8 ],
            vec![ 9, 0, 1 ]
        ]);
        matrix.map_mut(|_| 0);
        assert!(matrix.column_major_iter().all(|x| x == 0));
        matrix.map_mut_with_index(|_, r, c| r + c);
        assert_eq!(
            matrix,
            Matrix::from(vec![
                vec![ 0, 1, 2 ],
                vec![ 1, 2, 3 ],
                vec![ 2, 3, 4 ],
                vec![ 3, 4, 5 ]
            ])
        );
        assert_eq!(
            matrix,
            Matrix::from(vec![
                vec![ 9, 6, 5 ],
                vec![ 0, 2, 5 ],
                vec![ 5, 1, 2 ],
                vec![ 7, 7, 8 ]
            ]).map_with_index(|_, r, c| r + c)
        )
    }

    #[test]
    fn check_partition_quadrants() {
        let mut matrix = Matrix::from(vec![
            vec![ 0, 1, 2 ],
            vec![ 3, 4, 5 ],
            vec![ 6, 7, 8 ]
        ]);
        {
            let parts = matrix.partition_quadrants(2, 1);
            assert_eq!(parts.top_left, Matrix::column(vec![ 0, 3]));
            assert_eq!(parts.top_right, Matrix::from(vec![vec![ 1, 2 ], vec![ 4, 5]]));
            assert_eq!(parts.bottom_left, Matrix::column(vec![ 6 ]));
            assert_eq!(parts.bottom_right, Matrix::row(vec![ 7, 8]));
        }
        {
            let parts = matrix.partition_quadrants(1, 2);
            assert_eq!(parts.top_left, Matrix::row(vec![ 0, 1]));
            assert_eq!(parts.top_right, Matrix::column(vec![ 2 ]));
            assert_eq!(parts.bottom_left, Matrix::from(vec![vec![ 3, 4 ], vec![ 6, 7]]));
            assert_eq!(parts.bottom_right, Matrix::column(vec![ 5, 8]));
        }
        {
            let parts = matrix.partition_quadrants(0, 1);
            assert_eq!(parts.top_left.size(), (0, 0));
            assert_eq!(parts.top_right.size(), (0, 0));
            assert_eq!(parts.bottom_left, Matrix::column(vec![ 0, 3, 6 ]));
            assert_eq!(parts.bottom_right, Matrix::from(vec![vec![ 1, 2 ], vec![ 4, 5 ], vec![ 7, 8 ]]));
        }
        {
            let parts = matrix.partition_quadrants(2, 3);
            assert_eq!(parts.top_left, Matrix::from(vec![vec![ 0, 1, 2 ], vec![ 3, 4, 5 ]]));
            assert_eq!(parts.top_right.size(), (0, 0));
            assert_eq!(parts.bottom_left, Matrix::row(vec![ 6, 7, 8 ]));
            assert_eq!(parts.bottom_right.size(), (0, 0));
        }
    }

    #[test]
    fn check_general_partition() {
        use std::ops::Range;
        let mut matrix = Matrix::from_flat_row_major((10, 10), (0..100).collect());
        let partitions: [&[usize]; 6] = [
            &[ 3 ],
            &[ 2, 5, ],
            &[ 0, 10 ],
            &[ 0, 3, 10, ],
            &[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
            &[],
        ];
        let expected_slice_sizes: [&[Range<usize>]; 6] = [
            &[ 0..3, 3..10 ],
            &[ 0..2, 2..5, 5..10 ],
            &[ 0..0, 0..10, 10..10 ],
            &[ 0..0, 0..3, 3..10, 10..10 ],
            &[ 0..0, 0..1, 1..2, 2..3, 3..4, 4..5, 5..6, 6..7, 7..8, 8..9, 9..10, 10..10 ],
            &[ 0..10 ],
        ];
        for r in 0..6 {
            for c in 0..6 {
                let row_partitions = partitions[r];
                let column_partitions = partitions[c];
                let parts = matrix.partition(row_partitions, column_partitions);
                let expected_parts = (row_partitions.len() + 1) * (column_partitions.len() + 1);
                assert_eq!(expected_parts, parts.len());
                let expected_row_slices = expected_slice_sizes[r];
                let expected_column_slices = expected_slice_sizes[c];
                let parts_per_column = column_partitions.len() + 1;
                for (i, part) in parts.iter().enumerate() {
                    let (row_slice, column_slice) = (i / parts_per_column, i % parts_per_column);
                    let expected_slice = (
                        expected_row_slices[row_slice].clone(),
                        expected_column_slices[column_slice].clone()
                    );
                    let expected_size = (expected_slice.0.len(), expected_slice.1.len());
                    match expected_size {
                        (0, _) | (_, 0) => assert_eq!(part.size(), (0, 0)),
                        size => assert_eq!(part.size(), size),
                    };
                }
            }
        }
    }
}
