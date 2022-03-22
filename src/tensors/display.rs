use crate::tensors::views::TensorRef;

// Common formatting logic used for Tensor and TensorView Display implementations
pub(crate) fn format_view<T, S, const D: usize>(view: &S, f: &mut std::fmt::Formatter) -> std::fmt::Result
where
    T: std::fmt::Display,
    S: TensorRef<T, D>,
{
    // default to 3 decimals but allow the caller to override
    // TODO: ideally want to set significant figures instead of decimals
    let shape = view.view_shape();
    write!(f, "D = {:?}", D)?;
    if D > 0 {
        writeln!(f)?;
    }
    for d in 0..D {
        write!(f, "({:?}, {:?})", shape[d].0, shape[d].1)?;
        if d < D - 1 {
            write!(f, ", ")?;
        }
    }
    writeln!(f)?;
    match D {
        0 => {
            let value = match view.get_reference([0; D]) {
                Some(x) => x,
                None => panic!(
                    "Expected [] to be a valid index for {:?}",
                    shape
                ),
            };
            write!(
                f,
                "[ {:.*} ]",
                f.precision().unwrap_or(3),
                value
            )
        },
        1 => {
            write!(f, "[ ")?;
            let length = shape[0].1;
            for i in 0..length {
                let mut index = [0; D];
                index[0] = i;
                let value = match view.get_reference(index) {
                    Some(x) => x,
                    None => panic!(
                        "Expected {:?} to be a valid index for {:?}",
                        index,
                        shape
                    ),
                };
                write!(f, "{:.*}", f.precision().unwrap_or(3), value)?;
                if i < length - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, " ]")
        },
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
                        None => panic!(
                            "Expected {:?} to be a valid index for {:?}",
                            index,
                            shape
                        ),
                    };

                    write!(f, "{:.*}", f.precision().unwrap_or(3), value)?;
                    if column < columns - 1 {
                        write!(f, ", ")?;
                    }
                }
                if row < rows - 1 {
                    writeln!(f)?;
                }
            }
            write!(f, " ]")
        },
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
                            None => panic!(
                                "Expected {:?} to be a valid index for {:?}",
                                index,
                                shape
                            ),
                        };

                        write!(f, "{:.*}", f.precision().unwrap_or(3), value)?;
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
        },
        _d => {
            // TODO
            panic!("Dimensions greater than 3 are not yet supported for Display")
        }
    }
}

#[test]
fn test_display() {
    use crate::tensors::Tensor;
    let mut tensor_3 = Tensor::empty([("b", 3), ("x", 2), ("y", 2)], 0.0);
    tensor_3.map_mut_with_index(|[b,x,y], _| {
        (((y as i32) + (x as i32) * 2 + (b as i32) * 4) % 10) as f64
    });
    let mut tensor_2 = Tensor::empty([("x", 3), ("y", 4)], 0.0);
    tensor_2.map_mut_with_index(|[x,y], _| {
        (((y as i32) + (x as i32) * 4) % 10) as f64
    });
    let tensor_1 = Tensor::from([("x", 5)], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    let tensor_0 = Tensor::from_scalar(0.0);
    assert_eq!(tensor_3.to_string(), r#"D = 3
("b", 3), ("x", 2), ("y", 2)
[
  0.000, 1.000
  2.000, 3.000

  4.000, 5.000
  6.000, 7.000

  8.000, 9.000
  0.000, 1.000
]"#);
    assert_eq!(tensor_2.to_string(), r#"D = 2
("x", 3), ("y", 4)
[ 0.000, 1.000, 2.000, 3.000
  4.000, 5.000, 6.000, 7.000
  8.000, 9.000, 0.000, 1.000 ]"#);
    assert_eq!(tensor_1.to_string(), r#"D = 1
("x", 5)
[ 0.000, 1.000, 2.000, 3.000, 4.000 ]"#);
    assert_eq!(tensor_0.to_string(), r#"D = 0
[ 0.000 ]"#);
}
