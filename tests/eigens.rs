use easy_ml::matrices::Matrix;
use easy_ml::linear_algebra::eigens::{QRAlgorithm, FixedIterations, EigenvalueAlgorithm};

#[test]
fn test_simple_eigendecomposition() {
    let matrix: Matrix<f64> = Matrix::from(vec![
        vec![3.0, 0.0, 0.0],
        vec![4.0, 1.0, 0.0],
        vec![0.0, 0.0, 2.0],
    ]);
    let mut solver = QRAlgorithm::<f64, _>::new(FixedIterations::new(2000));
    let eigens = solver.solve(&matrix);
    assert!(eigens.is_ok());
    let eigens = eigens.unwrap();
    let absolute_difference: f64 = eigens.eigenvalues
        .iter()
        .cloned()
        .zip(vec![3.0, 1.0, 2.0].into_iter())
        .map(|(x, y)| (x - y).abs())
        .sum();
    assert!(absolute_difference < 0.0001);
    // let eigenvectors = eigens.eigenvectors;
    // println!("Eigenvectors\n{}", eigenvectors);
    // let eigenvalues = Matrix::from_diagonal(eigens.eigenvalues);
    // println!("Eigenvalues\n{}", eigenvalues);
    // let reconstruction = &eigenvectors * eigenvalues * eigenvectors.inverse().unwrap();
    // println!("Reconstructed input\n{}", reconstruction);
    // println!("Expected input\n{}", matrix);
    // let absolute_difference: f64 = matrix.column_major_iter()
    //     .zip(reconstruction.column_major_iter())
    //     .map(|(x, y)| (x - y).abs())
    //     .sum();
    // assert!(absolute_difference < 0.0001);
}
