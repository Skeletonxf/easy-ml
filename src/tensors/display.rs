use crate::tensors::indexing::TensorAccess;
use crate::tensors::views::TensorRef;

// Common formatting logic used for Tensor and TensorView Display implementations
pub(crate) fn format_view<T, S, const D: usize>(
    view: &S,
    f: &mut std::fmt::Formatter,
) -> std::fmt::Result
where
    T: std::fmt::Display,
    S: TensorRef<T, D>,
{
    let shape = view.view_shape();
    write!(f, "D = {:?}", D)?;
    if D > 0 {
        writeln!(f)?;
    }
    for (d, (name, length)) in shape.iter().enumerate() {
        write!(f, "({:?}, {:?})", name, length)?;
        if d < D - 1 {
            write!(f, ", ")?;
        }
    }
    writeln!(f)?;
    // It would be nice to default to some precision for f32 and f64 but I can't
    // work out how to easily check if T matches. If we use precision for all T
    // then strings get truncated which is even worse for debugging.
    match D {
        0 => {
            let value = match view.get_reference([0; D]) {
                Some(x) => x,
                None => panic!("Expected [] to be a valid index for {:?}", shape),
            };
            match f.precision() {
                Some(precision) => write!(f, "[ {:.*} ]", precision, value),
                None => write!(f, "[ {} ]", value),
            }
        }
        1 => {
            write!(f, "[ ")?;
            let length = shape[0].1;
            for i in 0..length {
                let mut index = [0; D];
                index[0] = i;
                let value = match view.get_reference(index) {
                    Some(x) => x,
                    None => panic!("Expected {:?} to be a valid index for {:?}", index, shape),
                };
                match f.precision() {
                    Some(precision) => write!(f, "{:.*}", precision, value)?,
                    None => write!(f, "{}", value)?,
                };
                if i < length - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, " ]")
        }
        2 => {
            write!(f, "[ ")?;
            let shape = view.view_shape();
            let rows = shape[0].1;
            let columns = shape[1].1;
            for row in 0..rows {
                if row > 0 {
                    write!(f, "  ")?;
                }
                for column in 0..columns {
                    let mut index = [0; D];
                    index[0] = row;
                    index[1] = column;
                    let value = match view.get_reference(index) {
                        Some(x) => x,
                        None => panic!("Expected {:?} to be a valid index for {:?}", index, shape),
                    };

                    match f.precision() {
                        Some(precision) => write!(f, "{:.*}", precision, value)?,
                        None => write!(f, "{}", value)?,
                    };
                    if column < columns - 1 {
                        write!(f, ", ")?;
                    }
                }
                if row < rows - 1 {
                    writeln!(f)?;
                }
            }
            write!(f, " ]")
        }
        3 => {
            writeln!(f, "[")?;
            let shape = view.view_shape();
            let blocks = shape[0].1;
            let rows = shape[1].1;
            let columns = shape[2].1;
            for block in 0..blocks {
                for row in 0..rows {
                    write!(f, "  ")?;
                    for column in 0..columns {
                        let mut index = [0; D];
                        index[0] = block;
                        index[1] = row;
                        index[2] = column;
                        let value = match view.get_reference(index) {
                            Some(x) => x,
                            None => {
                                panic!("Expected {:?} to be a valid index for {:?}", index, shape)
                            }
                        };

                        match f.precision() {
                            Some(precision) => write!(f, "{:.*}", precision, value)?,
                            None => write!(f, "{}", value)?,
                        };
                        if column < columns - 1 {
                            write!(f, ", ")?;
                        }
                    }
                    if row < rows - 1 {
                        writeln!(f)?;
                    }
                }
                if block < blocks - 1 {
                    writeln!(f)?;
                    writeln!(f)?;
                }
            }
            write!(f, "\n]")
        }
        n => {
            writeln!(f, "[")?;
            let shape = view.view_shape();
            let rows = shape[n - 2].1;
            let columns = shape[n - 1].1;
            let last_index = shape.map(|(_, l)| l - 1);
            for (index, value) in TensorAccess::from_source_order(view)
                .iter_reference()
                .with_index()
            {
                let row = index[n - 2];
                let column = index[n - 1];
                if column == 0 {
                    // starting a new row
                    write!(f, "  ")?;
                }
                match f.precision() {
                    Some(precision) => write!(f, "{:.*}", precision, value)?,
                    None => write!(f, "{}", value)?,
                };
                if column < columns - 1 {
                    write!(f, ", ")?;
                }
                // non final rows end with a newline, which happen when we're at the
                // end of a column index
                if row < rows - 1 && column == columns - 1 {
                    writeln!(f)?;
                }
                // the end of each block ends with a newline
                if row == rows - 1 && column == columns - 1 && index != last_index {
                    writeln!(f)?;
                    for dimension in (1..(n - 1)).rev() {
                        let index = index[dimension];
                        let length = shape[dimension].1;
                        // Each successive dimension we reach the end of is another newline
                        // because the next value will increment the left-er dimension by 1
                        // This means a 5 dimensional tensor will have a 3 line gap between the
                        // leftmost dimension increments, the second dimension gets 2 line gaps,
                        // and the third dimension gets 1 line gaps with the fourth and fifth
                        // dimensions being shown in row/column blocks
                        if index == length - 1 {
                            writeln!(f)?;
                        } else {
                            break;
                        }
                    }
                }
            }
            write!(f, "\n]")
        }
    }
}

#[test]
fn test_display() {
    use crate::tensors::Tensor;
    #[rustfmt::skip]
    let tensor_3 = Tensor::empty([("b", 3), ("x", 2), ("y", 2)], 0.0)
        .map_with_index(|[b, x, y], _| {
            (((y as i32) + (x as i32) * 2 + (b as i32) * 4) % 10) as f64
        });
    let tensor_2 = Tensor::empty([("x", 3), ("y", 4)], 0.0)
        .map_with_index(|[x, y], _| (((y as i32) + (x as i32) * 4) % 10) as f64);
    let tensor_1 = Tensor::from([("x", 5)], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    let tensor_0 = Tensor::from_scalar(0.0);
    assert_eq!(
        format!("{:.3}", tensor_3),
        r#"D = 3
("b", 3), ("x", 2), ("y", 2)
[
  0.000, 1.000
  2.000, 3.000

  4.000, 5.000
  6.000, 7.000

  8.000, 9.000
  0.000, 1.000
]"#
    );
    assert_eq!(
        format!("{:.3}", tensor_2),
        r#"D = 2
("x", 3), ("y", 4)
[ 0.000, 1.000, 2.000, 3.000
  4.000, 5.000, 6.000, 7.000
  8.000, 9.000, 0.000, 1.000 ]"#
    );
    assert_eq!(
        format!("{:.3}", tensor_1),
        r#"D = 1
("x", 5)
[ 0.000, 1.000, 2.000, 3.000, 4.000 ]"#
    );
    assert_eq!(
        format!("{:.3}", tensor_0),
        r#"D = 0
[ 0.000 ]"#
    );

    assert_eq!(
        format!("{:.3}", tensor_3.index_by(["x", "y", "b"])),
        r#"D = 3
("x", 2), ("y", 2), ("b", 3)
[
  0.000, 4.000, 8.000
  1.000, 5.000, 9.000

  2.000, 6.000, 0.000
  3.000, 7.000, 1.000
]
Data Layout = Linear(["b", "x", "y"])"#
    );

    println!("{:?}", tensor_3.transpose_view(["x", "y", "b"]).source());
    assert_eq!(
        format!("{:.3}", tensor_3.transpose_view(["x", "y", "b"]).source()),
        r#"D = 3
("b", 2), ("x", 2), ("y", 3)
[
  0.000, 4.000, 8.000
  1.000, 5.000, 9.000

  2.000, 6.000, 0.000
  3.000, 7.000, 1.000
]
Data Layout = Linear(["y", "b", "x"])"#
    );
}

#[test]
fn test_display_large_dimensionality() {
    use crate::tensors::Tensor;
    let tensor_5 = Tensor::from(
        [("a", 2), ("b", 2), ("c", 2), ("d", 2), ("e", 2)],
        (0..10).cycle().take(2 * 2 * 2 * 2 * 2).collect(),
    );
    assert_eq!(
        format!("{:.3}", tensor_5),
        r#"D = 5
("a", 2), ("b", 2), ("c", 2), ("d", 2), ("e", 2)
[
  0, 1
  2, 3

  4, 5
  6, 7


  8, 9
  0, 1

  2, 3
  4, 5



  6, 7
  8, 9

  0, 1
  2, 3


  4, 5
  6, 7

  8, 9
  0, 1
]"#
    );
}
