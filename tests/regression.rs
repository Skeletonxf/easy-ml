#![allow(non_snake_case)]

extern crate easy_ml;

/// A simple linear regression acceptance test / example.

#[cfg(test)]
mod tests {
    use easy_ml::matrices::Matrix;
    use easy_ml::linear_algebra;

    #[test]
    fn linear_regression() {
        // First create some data to perform regression on
        let x = Matrix::column(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let y = Matrix::column(
            vec![1.0, 3.0, 3.5, 8.0, 9.0, 11.0, 13.0, 15.5, 17.5, 19.0, 21.0, 23.0, 25.0]);
        // Try to fit a line y = mx + c

        // We will compute w as a column vector with values corresponding to c then m

        // Define the design matrix consisting of 1 and each x in each row
        // TODO: Some stacking methods like in numpy would be useful here
        let mut X = Matrix::empty(1.0, (x.rows(), 1 + x.columns()));
        for i in 0..X.rows() {
            for j in 0..X.columns() {
                if j == 0 {
                    continue
                }
                X.set(i, j, x.get(i, 0));
            }
        }
        println!("{:?}", &X);

        // w is given by inverse(XT*X) * (XT * y)
        let w = linear_algebra::inverse(&(X.transpose() * &X)).unwrap() * (X.transpose() * &y);
        let error = error_function(&w, &X, &y);
        println!("error {:?}", error);
        println!("y = {:?}\nprediction = {:?}", y, (&X * &w));
        assert!(error < 3.7);
        assert!(error > 3.5);
    }

    /*
     * A sum of squares error function. Xw computes each ith prediction as a column vector
     */
    fn error_function(w: &Matrix<f32>, X: &Matrix<f32>, y: &Matrix<f32>) -> f32 {
        let error = y - (X * w);
        (error.transpose() * error).get(0, 0)
    }

    // TODO: turn this into a test
    // // Inversing an arbitarty square matrix is hard. We only need to inverse a 2x2 here
    // // TODO: See if there's an algorithm that can be implemented for arbitary inversions
    // fn invese(matrix: Matrix<f32>) -> Matrix<f32> {
    //     let a = matrix.get(0, 0);
    //     let b = matrix.get(0, 1);
    //     let c = matrix.get(1, 0);
    //     let d = matrix.get(1, 1);
    //     let determinant = (a * d) - (b * c);
    //     Matrix::from(vec![
    //         vec![ d, -b ],
    //         vec![ -c, a ]
    //     ]).map(|x| x / determinant)
    // }
}
