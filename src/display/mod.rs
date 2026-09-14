use crate::matrices::views::MatrixRef;
use crate::tensors::dimensions::elements;
use crate::tensors::indexing::TensorAccess;
use crate::tensors::views::TensorRef;

/**
 * Configuration options for the summary logic in matrices and tensors.
 *
 * You can pass a [SummaryOptions] to a Matrix or Tensor to customise the
 * formatting via [display_with](crate::matrices::Matrix::display_with) to
 * create a wrapper type that uses the custom options or
 * [fmt_with](crate::matrices::Matrix::fmt_with) for direct formatting.
 *
 * The default Display implementation for matrices and tensors uses the
 * default SummaryOptions, with 1000 elements as the threshold for summarisation
 * and 3 items shown along each dimension.
 *
 * ```
 * use easy_ml::matrices::{Matrix, SummaryOptions};
 * let config = SummaryOptions::default()
 *    .with_threshold_of(Some(1000))
 *    .with_summary_items_of(3);
 * let matrix = Matrix::column(vec![ 0, 1, 2, 3 ]);
 * println!("{}", matrix.display_with(config));
 * ```
 */
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SummaryOptions {
    threshold_number_of_elements: Option<usize>,
    summary_items_in_each_dimension: usize,
}

impl SummaryOptions {
    /**
     * Changes the configured threshold for how many elements in the matrix/tensor should
     * be displayed in full before summarising instead. None means the display will never
     * be summarised.
     */
    pub fn with_threshold_of(self, threshold_number_of_elements: Option<usize>) -> Self {
        SummaryOptions {
            threshold_number_of_elements: threshold_number_of_elements,
            summary_items_in_each_dimension: self.summary_items_in_each_dimension,
        }
    }

    /**
     * Changes the configured threshold for how many elements in each dimension should
     * be shown when summarising.
     */
    pub fn with_summary_items_of(self, summary_items_in_each_dimension: usize) -> Self {
        SummaryOptions {
            threshold_number_of_elements: self.threshold_number_of_elements,
            summary_items_in_each_dimension: summary_items_in_each_dimension,
        }
    }

    /**
     * Returns the configured threshold for how many elements in the matrix/tensor should
     * be displayed in full before summarising instead. None means the display will never
     * be summarised.
     */
    pub fn threshold(&self) -> Option<usize> {
        self.threshold_number_of_elements
    }

    /**
     * Returns the configured threshold for how many elements in each dimension should
     * be shown when summarising.
     */
    pub fn summary_items(&self) -> usize {
        self.summary_items_in_each_dimension
    }

    fn is_over_threshold(&self, total_elements: usize) -> bool {
        match self.threshold_number_of_elements {
            None => false,
            Some(threshold) => total_elements >= threshold,
        }
    }
}

impl Default for SummaryOptions {
    /**
     * Returns the default configuration of a threshold of 1000 elements and 3 summary items.
     */
    fn default() -> Self {
        SummaryOptions {
            threshold_number_of_elements: Some(1000),
            summary_items_in_each_dimension: 3,
        }
    }
}

// Common formatting logic used for Matrix and MatrixView Display implementations
pub(crate) fn format_view<T, S>(
    view: &S,
    f: &mut std::fmt::Formatter,
    configuration: SummaryOptions,
) -> std::fmt::Result
where
    T: std::fmt::Display,
    S: MatrixRef<T>,
{
    let rows = view.view_rows();
    let columns = view.view_columns();
    // It would be nice to default to some precision for f32 and f64 but I can't
    // work out how to easily check if T matches. If we use precision for all T
    // then strings get truncated which is even worse for debugging.
    let summarize = configuration.is_over_threshold(rows * columns);
    write!(f, "[ ")?;
    for row in 0..rows {
        if summarize {
            let start = configuration.summary_items();
            let end = rows.saturating_sub(configuration.summary_items());
            let skip_row = row >= start && row < end;
            if skip_row {
                // Skip all the rows for the middle of the matrix not within
                // the summary item range, but only insert the ellipsis once
                // at the start of the skipped rows.
                if row == start {
                    writeln!(f, "  ...")?;
                }
                continue;
            }
        }
        if row > 0 {
            write!(f, "  ")?;
        }
        for column in 0..columns {
            if summarize {
                let start = configuration.summary_items();
                let end = columns.saturating_sub(configuration.summary_items());
                let skip_column = column >= start && column < end;
                if skip_column {
                    // Skip all the columns for the middle of the matrix not within
                    // the summary item range, but only insert the ellipsis once
                    // at the start of the skipped columns.
                    if column == start {
                        write!(f, "..., ")?;
                    }
                    continue;
                }
            }
            let value = match view.try_get_reference(row, column) {
                Some(x) => x,
                None => panic!(
                    "Expected ({},{}) to be a valid index in range of (0,0) to ({},{})",
                    row,
                    column,
                    rows - 1,
                    columns - 1
                ),
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

#[test]
fn test_summary_output_matrix() {
    use crate::matrices::Matrix;
    let matrix = Matrix::from_fn((25, 25), |(r, c)| r * 25 + c);
    let formatted = format!(
        "{}",
        matrix.display_with(SummaryOptions::default().with_threshold_of(Some(100)))
    );
    assert_eq!(
        formatted,
        r#"[ 0, 1, 2, ..., 22, 23, 24
  25, 26, 27, ..., 47, 48, 49
  50, 51, 52, ..., 72, 73, 74
  ...
  550, 551, 552, ..., 572, 573, 574
  575, 576, 577, ..., 597, 598, 599
  600, 601, 602, ..., 622, 623, 624 ]"#
    )
}

#[test]
fn test_summary_output_matrix_view() {
    use crate::matrices::Matrix;
    use crate::matrices::views::MatrixView;
    let matrix = MatrixView::from(Matrix::from_fn((25, 25), |(r, c)| r * 25 + c));
    let formatted = format!(
        "{}",
        matrix.display_with(SummaryOptions::default().with_threshold_of(Some(625)))
    );
    assert_eq!(
        formatted,
        r#"[ 0, 1, 2, ..., 22, 23, 24
  25, 26, 27, ..., 47, 48, 49
  50, 51, 52, ..., 72, 73, 74
  ...
  550, 551, 552, ..., 572, 573, 574
  575, 576, 577, ..., 597, 598, 599
  600, 601, 602, ..., 622, 623, 624 ]"#
    )
}

// Common formatting logic used for Tensor and TensorView Display implementations
pub(crate) fn format_view_tensor<T, S, const D: usize>(
    view: &S,
    f: &mut std::fmt::Formatter,
    configuration: SummaryOptions,
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
    let summarize = configuration.is_over_threshold(elements(&shape));
    match D {
        0 => {
            let value = match view.get_reference([0; D]) {
                Some(x) => x,
                None => panic!("Expected [] to be a valid index for {:?}", shape),
            };
            // If the configuration has a threshold of zero and 0 summary items
            // we should 'summarise' this single element. For any other configuration
            // the D=0 case can't need a summary.
            if configuration.threshold() == Some(0) && configuration.summary_items() == 0 {
                write!(f, "[ ... ]")
            } else {
                match f.precision() {
                    Some(precision) => write!(f, "[ {:.*} ]", precision, value),
                    None => write!(f, "[ {} ]", value),
                }
            }
        }
        1 => {
            write!(f, "[ ")?;
            let length = shape[0].1;
            for i in 0..length {
                let mut index = [0; D];
                index[0] = i;

                if summarize {
                    let start = configuration.summary_items();
                    let end = length.saturating_sub(configuration.summary_items());
                    let skip_item = i >= start && i < end;
                    if skip_item {
                        // Skip all the items for the middle of the vector not within
                        // the summary item range, but only insert the ellipsis once
                        // at the start of the skipped items.
                        if i == start {
                            write!(f, "..., ")?;
                        }
                        continue;
                    }
                }

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
                if summarize {
                    let start = configuration.summary_items();
                    let end = rows.saturating_sub(configuration.summary_items());
                    let skip_row = row >= start && row < end;
                    if skip_row {
                        // Skip all the rows for the middle of the tensor not within
                        // the summary item range, but only insert the ellipsis once
                        // at the start of the skipped rows.
                        if row == start {
                            writeln!(f, "  ...")?;
                        }
                        continue;
                    }
                }
                if row > 0 {
                    write!(f, "  ")?;
                }
                for column in 0..columns {
                    if summarize {
                        let start = configuration.summary_items();
                        let end = columns.saturating_sub(configuration.summary_items());
                        let skip_column = column >= start && column < end;
                        if skip_column {
                            // Skip all the columns for the middle of the tensor not within
                            // the summary item range, but only insert the ellipsis once
                            // at the start of the skipped columns.
                            if column == start {
                                write!(f, "..., ")?;
                            }
                            continue;
                        }
                    }

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

#[test]
fn test_summary_output_tensor_0_dimensions() {
    use crate::tensors::Tensor;
    let scalar = Tensor::from_scalar(3);
    let formatted = format!(
        "{}",
        scalar.display_with(SummaryOptions::default().with_threshold_of(Some(0)).with_summary_items_of(0))
    );
    assert_eq!(
        formatted,
        r#"D = 0
[ ... ]"#
    )
}

#[test]
fn test_summary_output_tensor_1_dimension() {
    use crate::tensors::Tensor;
    let vector = Tensor::from_fn([("item", 25)], |[i]| i);
    let formatted = format!(
        "{}",
        vector.display_with(SummaryOptions::default().with_threshold_of(Some(25)))
    );
    assert_eq!(
        formatted,
        r#"D = 1
("item", 25)
[ 0, 1, 2, ..., 22, 23, 24 ]"#
    )
}

#[test]
fn test_summary_output_tensor_2_dimensions() {
    use crate::tensors::Tensor;
    let matrix = Tensor::from_fn([("r", 25), ("c", 25)], |[r, c]| r * 25 + c);
    let formatted = format!(
        "{}",
        matrix.display_with(SummaryOptions::default().with_threshold_of(Some(600)))
    );
    assert_eq!(
        formatted,
        r#"D = 2
("r", 25), ("c", 25)
[ 0, 1, 2, ..., 22, 23, 24
  25, 26, 27, ..., 47, 48, 49
  50, 51, 52, ..., 72, 73, 74
  ...
  550, 551, 552, ..., 572, 573, 574
  575, 576, 577, ..., 597, 598, 599
  600, 601, 602, ..., 622, 623, 624 ]"#
    )
}
