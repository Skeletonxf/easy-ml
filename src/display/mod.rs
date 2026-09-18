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
                if summarize {
                    let start = configuration.summary_items();
                    let end = blocks.saturating_sub(configuration.summary_items());
                    let skip_block = block >= start && block < end;
                    if skip_block {
                        // Skip all the blocks for the middle of the tensor not within
                        // the summary item range, but only insert the ellipsis once
                        // at the start of the skipped blocks.
                        // We do a double line of ellipsis here to distinguish skipped
                        // blocks from skipped rows.
                        if block == start {
                            writeln!(f, "  ...\n  ...\n")?;
                        }
                        continue;
                    }
                }
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
                                writeln!(f, "  ...,")?;
                            }
                            continue;
                        }
                    }
                    write!(f, "  ")?;
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
            'elements: for (index, value) in TensorAccess::from_source_order(view)
                .iter_reference()
                .with_index()
            {
                let row = index[n - 2];
                let column = index[n - 1];

                if summarize {
                    let start = configuration.summary_items();
                    for dimension in 0..(n - 2) {
                        let i = index[dimension];
                        let length = shape[dimension].1;
                        let end = length.saturating_sub(configuration.summary_items());
                        let skip_dimension = i >= start && i < end;
                        if skip_dimension {
                            // Skip all this dimension for the middle of the tensor not within
                            // the summary item range, but only insert the ellipsis block once
                            // at the start of the skipped blocks.
                            // We do a multi line of ellipsis here to distinguish which
                            // dimension is skipped.
                            if i == start {
                                let mut smaller_dimensions_at_0 = true;
                                for dimension in (dimension + 1)..n {
                                    smaller_dimensions_at_0 = smaller_dimensions_at_0 && index[dimension] == 0;
                                }
                                if smaller_dimensions_at_0 {
                                    let repeats = (shape.len() - dimension) - 1;
                                    for _ in 0..repeats {
                                        writeln!(f, "  ...")?;
                                    }
                                    for _ in 0..(repeats - 1) {
                                        writeln!(f, "")?;
                                    }
                                }
                            }
                            continue 'elements;
                        }
                    }
                    {
                        let end = rows.saturating_sub(configuration.summary_items());
                        let skip_row = row >= start && row < end;
                        if skip_row {
                            // Skip all the rows for the middle of the tensor not within
                            // the summary item range, but only insert the ellipsis once
                            // at the start of the skipped rows.
                            if row == start && column == 0 {
                                writeln!(f, "  ...,")?;
                            }
                            continue;
                        }
                    }
                    {
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
                }

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
fn test_display_large_dimensionality_uneven() {
    use crate::tensors::Tensor;
    let tensor_5 = Tensor::from(
        [("a", 2), ("b", 4), ("c", 3), ("d", 2), ("e", 2)],
        (0..10).cycle().take(2 * 4 * 3 * 2 * 2).collect(),
    );
    assert_eq!(
        format!("{:.3}", tensor_5),
        r#"D = 5
("a", 2), ("b", 4), ("c", 3), ("d", 2), ("e", 2)
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

  2, 3
  4, 5
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

#[test]
fn test_summary_output_tensor_3_dimensions() {
    use crate::tensors::Tensor;
    let tensor = Tensor::from_fn([("b", 7), ("r", 8), ("c", 9)], |[b, r, c]| (b * (8 * 9)) + (r * 9) + c);
    let formatted = format!(
        "{}",
        tensor.display_with(SummaryOptions::default().with_threshold_of(Some(500)))
    );
    assert_eq!(
        formatted,
        r#"D = 3
("b", 7), ("r", 8), ("c", 9)
[
  0, 1, 2, ..., 6, 7, 8
  9, 10, 11, ..., 15, 16, 17
  18, 19, 20, ..., 24, 25, 26
  ...,
  45, 46, 47, ..., 51, 52, 53
  54, 55, 56, ..., 60, 61, 62
  63, 64, 65, ..., 69, 70, 71

  72, 73, 74, ..., 78, 79, 80
  81, 82, 83, ..., 87, 88, 89
  90, 91, 92, ..., 96, 97, 98
  ...,
  117, 118, 119, ..., 123, 124, 125
  126, 127, 128, ..., 132, 133, 134
  135, 136, 137, ..., 141, 142, 143

  144, 145, 146, ..., 150, 151, 152
  153, 154, 155, ..., 159, 160, 161
  162, 163, 164, ..., 168, 169, 170
  ...,
  189, 190, 191, ..., 195, 196, 197
  198, 199, 200, ..., 204, 205, 206
  207, 208, 209, ..., 213, 214, 215

  ...
  ...

  288, 289, 290, ..., 294, 295, 296
  297, 298, 299, ..., 303, 304, 305
  306, 307, 308, ..., 312, 313, 314
  ...,
  333, 334, 335, ..., 339, 340, 341
  342, 343, 344, ..., 348, 349, 350
  351, 352, 353, ..., 357, 358, 359

  360, 361, 362, ..., 366, 367, 368
  369, 370, 371, ..., 375, 376, 377
  378, 379, 380, ..., 384, 385, 386
  ...,
  405, 406, 407, ..., 411, 412, 413
  414, 415, 416, ..., 420, 421, 422
  423, 424, 425, ..., 429, 430, 431

  432, 433, 434, ..., 438, 439, 440
  441, 442, 443, ..., 447, 448, 449
  450, 451, 452, ..., 456, 457, 458
  ...,
  477, 478, 479, ..., 483, 484, 485
  486, 487, 488, ..., 492, 493, 494
  495, 496, 497, ..., 501, 502, 503
]"#)
}

#[test]
fn test_summary_output_tensor_5_dimensions() {
    use crate::tensors::Tensor;
    let tensor = Tensor::from_fn(
        [("b", 2), ("r", 8), ("c", 7), ("w", 7), ("h", 8)],
        |[b, r, c, w, h]| ((b * 3136) + (r * 392) + (c * 56) + (w * 8) + h) as i16);
    let formatted = format!("{}", tensor);
    assert_eq!(
        formatted,
        r#"D = 5
("b", 2), ("r", 8), ("c", 7), ("w", 7), ("h", 8)
[
  0, 1, 2, ..., 5, 6, 7
  8, 9, 10, ..., 13, 14, 15
  16, 17, 18, ..., 21, 22, 23
  ...,
  32, 33, 34, ..., 37, 38, 39
  40, 41, 42, ..., 45, 46, 47
  48, 49, 50, ..., 53, 54, 55

  56, 57, 58, ..., 61, 62, 63
  64, 65, 66, ..., 69, 70, 71
  72, 73, 74, ..., 77, 78, 79
  ...,
  88, 89, 90, ..., 93, 94, 95
  96, 97, 98, ..., 101, 102, 103
  104, 105, 106, ..., 109, 110, 111

  112, 113, 114, ..., 117, 118, 119
  120, 121, 122, ..., 125, 126, 127
  128, 129, 130, ..., 133, 134, 135
  ...,
  144, 145, 146, ..., 149, 150, 151
  152, 153, 154, ..., 157, 158, 159
  160, 161, 162, ..., 165, 166, 167

  ...
  ...

  224, 225, 226, ..., 229, 230, 231
  232, 233, 234, ..., 237, 238, 239
  240, 241, 242, ..., 245, 246, 247
  ...,
  256, 257, 258, ..., 261, 262, 263
  264, 265, 266, ..., 269, 270, 271
  272, 273, 274, ..., 277, 278, 279

  280, 281, 282, ..., 285, 286, 287
  288, 289, 290, ..., 293, 294, 295
  296, 297, 298, ..., 301, 302, 303
  ...,
  312, 313, 314, ..., 317, 318, 319
  320, 321, 322, ..., 325, 326, 327
  328, 329, 330, ..., 333, 334, 335

  336, 337, 338, ..., 341, 342, 343
  344, 345, 346, ..., 349, 350, 351
  352, 353, 354, ..., 357, 358, 359
  ...,
  368, 369, 370, ..., 373, 374, 375
  376, 377, 378, ..., 381, 382, 383
  384, 385, 386, ..., 389, 390, 391


  392, 393, 394, ..., 397, 398, 399
  400, 401, 402, ..., 405, 406, 407
  408, 409, 410, ..., 413, 414, 415
  ...,
  424, 425, 426, ..., 429, 430, 431
  432, 433, 434, ..., 437, 438, 439
  440, 441, 442, ..., 445, 446, 447

  448, 449, 450, ..., 453, 454, 455
  456, 457, 458, ..., 461, 462, 463
  464, 465, 466, ..., 469, 470, 471
  ...,
  480, 481, 482, ..., 485, 486, 487
  488, 489, 490, ..., 493, 494, 495
  496, 497, 498, ..., 501, 502, 503

  504, 505, 506, ..., 509, 510, 511
  512, 513, 514, ..., 517, 518, 519
  520, 521, 522, ..., 525, 526, 527
  ...,
  536, 537, 538, ..., 541, 542, 543
  544, 545, 546, ..., 549, 550, 551
  552, 553, 554, ..., 557, 558, 559

  ...
  ...

  616, 617, 618, ..., 621, 622, 623
  624, 625, 626, ..., 629, 630, 631
  632, 633, 634, ..., 637, 638, 639
  ...,
  648, 649, 650, ..., 653, 654, 655
  656, 657, 658, ..., 661, 662, 663
  664, 665, 666, ..., 669, 670, 671

  672, 673, 674, ..., 677, 678, 679
  680, 681, 682, ..., 685, 686, 687
  688, 689, 690, ..., 693, 694, 695
  ...,
  704, 705, 706, ..., 709, 710, 711
  712, 713, 714, ..., 717, 718, 719
  720, 721, 722, ..., 725, 726, 727

  728, 729, 730, ..., 733, 734, 735
  736, 737, 738, ..., 741, 742, 743
  744, 745, 746, ..., 749, 750, 751
  ...,
  760, 761, 762, ..., 765, 766, 767
  768, 769, 770, ..., 773, 774, 775
  776, 777, 778, ..., 781, 782, 783


  784, 785, 786, ..., 789, 790, 791
  792, 793, 794, ..., 797, 798, 799
  800, 801, 802, ..., 805, 806, 807
  ...,
  816, 817, 818, ..., 821, 822, 823
  824, 825, 826, ..., 829, 830, 831
  832, 833, 834, ..., 837, 838, 839

  840, 841, 842, ..., 845, 846, 847
  848, 849, 850, ..., 853, 854, 855
  856, 857, 858, ..., 861, 862, 863
  ...,
  872, 873, 874, ..., 877, 878, 879
  880, 881, 882, ..., 885, 886, 887
  888, 889, 890, ..., 893, 894, 895

  896, 897, 898, ..., 901, 902, 903
  904, 905, 906, ..., 909, 910, 911
  912, 913, 914, ..., 917, 918, 919
  ...,
  928, 929, 930, ..., 933, 934, 935
  936, 937, 938, ..., 941, 942, 943
  944, 945, 946, ..., 949, 950, 951

  ...
  ...

  1008, 1009, 1010, ..., 1013, 1014, 1015
  1016, 1017, 1018, ..., 1021, 1022, 1023
  1024, 1025, 1026, ..., 1029, 1030, 1031
  ...,
  1040, 1041, 1042, ..., 1045, 1046, 1047
  1048, 1049, 1050, ..., 1053, 1054, 1055
  1056, 1057, 1058, ..., 1061, 1062, 1063

  1064, 1065, 1066, ..., 1069, 1070, 1071
  1072, 1073, 1074, ..., 1077, 1078, 1079
  1080, 1081, 1082, ..., 1085, 1086, 1087
  ...,
  1096, 1097, 1098, ..., 1101, 1102, 1103
  1104, 1105, 1106, ..., 1109, 1110, 1111
  1112, 1113, 1114, ..., 1117, 1118, 1119

  1120, 1121, 1122, ..., 1125, 1126, 1127
  1128, 1129, 1130, ..., 1133, 1134, 1135
  1136, 1137, 1138, ..., 1141, 1142, 1143
  ...,
  1152, 1153, 1154, ..., 1157, 1158, 1159
  1160, 1161, 1162, ..., 1165, 1166, 1167
  1168, 1169, 1170, ..., 1173, 1174, 1175


  ...
  ...
  ...


  1960, 1961, 1962, ..., 1965, 1966, 1967
  1968, 1969, 1970, ..., 1973, 1974, 1975
  1976, 1977, 1978, ..., 1981, 1982, 1983
  ...,
  1992, 1993, 1994, ..., 1997, 1998, 1999
  2000, 2001, 2002, ..., 2005, 2006, 2007
  2008, 2009, 2010, ..., 2013, 2014, 2015

  2016, 2017, 2018, ..., 2021, 2022, 2023
  2024, 2025, 2026, ..., 2029, 2030, 2031
  2032, 2033, 2034, ..., 2037, 2038, 2039
  ...,
  2048, 2049, 2050, ..., 2053, 2054, 2055
  2056, 2057, 2058, ..., 2061, 2062, 2063
  2064, 2065, 2066, ..., 2069, 2070, 2071

  2072, 2073, 2074, ..., 2077, 2078, 2079
  2080, 2081, 2082, ..., 2085, 2086, 2087
  2088, 2089, 2090, ..., 2093, 2094, 2095
  ...,
  2104, 2105, 2106, ..., 2109, 2110, 2111
  2112, 2113, 2114, ..., 2117, 2118, 2119
  2120, 2121, 2122, ..., 2125, 2126, 2127

  ...
  ...

  2184, 2185, 2186, ..., 2189, 2190, 2191
  2192, 2193, 2194, ..., 2197, 2198, 2199
  2200, 2201, 2202, ..., 2205, 2206, 2207
  ...,
  2216, 2217, 2218, ..., 2221, 2222, 2223
  2224, 2225, 2226, ..., 2229, 2230, 2231
  2232, 2233, 2234, ..., 2237, 2238, 2239

  2240, 2241, 2242, ..., 2245, 2246, 2247
  2248, 2249, 2250, ..., 2253, 2254, 2255
  2256, 2257, 2258, ..., 2261, 2262, 2263
  ...,
  2272, 2273, 2274, ..., 2277, 2278, 2279
  2280, 2281, 2282, ..., 2285, 2286, 2287
  2288, 2289, 2290, ..., 2293, 2294, 2295

  2296, 2297, 2298, ..., 2301, 2302, 2303
  2304, 2305, 2306, ..., 2309, 2310, 2311
  2312, 2313, 2314, ..., 2317, 2318, 2319
  ...,
  2328, 2329, 2330, ..., 2333, 2334, 2335
  2336, 2337, 2338, ..., 2341, 2342, 2343
  2344, 2345, 2346, ..., 2349, 2350, 2351


  2352, 2353, 2354, ..., 2357, 2358, 2359
  2360, 2361, 2362, ..., 2365, 2366, 2367
  2368, 2369, 2370, ..., 2373, 2374, 2375
  ...,
  2384, 2385, 2386, ..., 2389, 2390, 2391
  2392, 2393, 2394, ..., 2397, 2398, 2399
  2400, 2401, 2402, ..., 2405, 2406, 2407

  2408, 2409, 2410, ..., 2413, 2414, 2415
  2416, 2417, 2418, ..., 2421, 2422, 2423
  2424, 2425, 2426, ..., 2429, 2430, 2431
  ...,
  2440, 2441, 2442, ..., 2445, 2446, 2447
  2448, 2449, 2450, ..., 2453, 2454, 2455
  2456, 2457, 2458, ..., 2461, 2462, 2463

  2464, 2465, 2466, ..., 2469, 2470, 2471
  2472, 2473, 2474, ..., 2477, 2478, 2479
  2480, 2481, 2482, ..., 2485, 2486, 2487
  ...,
  2496, 2497, 2498, ..., 2501, 2502, 2503
  2504, 2505, 2506, ..., 2509, 2510, 2511
  2512, 2513, 2514, ..., 2517, 2518, 2519

  ...
  ...

  2576, 2577, 2578, ..., 2581, 2582, 2583
  2584, 2585, 2586, ..., 2589, 2590, 2591
  2592, 2593, 2594, ..., 2597, 2598, 2599
  ...,
  2608, 2609, 2610, ..., 2613, 2614, 2615
  2616, 2617, 2618, ..., 2621, 2622, 2623
  2624, 2625, 2626, ..., 2629, 2630, 2631

  2632, 2633, 2634, ..., 2637, 2638, 2639
  2640, 2641, 2642, ..., 2645, 2646, 2647
  2648, 2649, 2650, ..., 2653, 2654, 2655
  ...,
  2664, 2665, 2666, ..., 2669, 2670, 2671
  2672, 2673, 2674, ..., 2677, 2678, 2679
  2680, 2681, 2682, ..., 2685, 2686, 2687

  2688, 2689, 2690, ..., 2693, 2694, 2695
  2696, 2697, 2698, ..., 2701, 2702, 2703
  2704, 2705, 2706, ..., 2709, 2710, 2711
  ...,
  2720, 2721, 2722, ..., 2725, 2726, 2727
  2728, 2729, 2730, ..., 2733, 2734, 2735
  2736, 2737, 2738, ..., 2741, 2742, 2743


  2744, 2745, 2746, ..., 2749, 2750, 2751
  2752, 2753, 2754, ..., 2757, 2758, 2759
  2760, 2761, 2762, ..., 2765, 2766, 2767
  ...,
  2776, 2777, 2778, ..., 2781, 2782, 2783
  2784, 2785, 2786, ..., 2789, 2790, 2791
  2792, 2793, 2794, ..., 2797, 2798, 2799

  2800, 2801, 2802, ..., 2805, 2806, 2807
  2808, 2809, 2810, ..., 2813, 2814, 2815
  2816, 2817, 2818, ..., 2821, 2822, 2823
  ...,
  2832, 2833, 2834, ..., 2837, 2838, 2839
  2840, 2841, 2842, ..., 2845, 2846, 2847
  2848, 2849, 2850, ..., 2853, 2854, 2855

  2856, 2857, 2858, ..., 2861, 2862, 2863
  2864, 2865, 2866, ..., 2869, 2870, 2871
  2872, 2873, 2874, ..., 2877, 2878, 2879
  ...,
  2888, 2889, 2890, ..., 2893, 2894, 2895
  2896, 2897, 2898, ..., 2901, 2902, 2903
  2904, 2905, 2906, ..., 2909, 2910, 2911

  ...
  ...

  2968, 2969, 2970, ..., 2973, 2974, 2975
  2976, 2977, 2978, ..., 2981, 2982, 2983
  2984, 2985, 2986, ..., 2989, 2990, 2991
  ...,
  3000, 3001, 3002, ..., 3005, 3006, 3007
  3008, 3009, 3010, ..., 3013, 3014, 3015
  3016, 3017, 3018, ..., 3021, 3022, 3023

  3024, 3025, 3026, ..., 3029, 3030, 3031
  3032, 3033, 3034, ..., 3037, 3038, 3039
  3040, 3041, 3042, ..., 3045, 3046, 3047
  ...,
  3056, 3057, 3058, ..., 3061, 3062, 3063
  3064, 3065, 3066, ..., 3069, 3070, 3071
  3072, 3073, 3074, ..., 3077, 3078, 3079

  3080, 3081, 3082, ..., 3085, 3086, 3087
  3088, 3089, 3090, ..., 3093, 3094, 3095
  3096, 3097, 3098, ..., 3101, 3102, 3103
  ...,
  3112, 3113, 3114, ..., 3117, 3118, 3119
  3120, 3121, 3122, ..., 3125, 3126, 3127
  3128, 3129, 3130, ..., 3133, 3134, 3135



  3136, 3137, 3138, ..., 3141, 3142, 3143
  3144, 3145, 3146, ..., 3149, 3150, 3151
  3152, 3153, 3154, ..., 3157, 3158, 3159
  ...,
  3168, 3169, 3170, ..., 3173, 3174, 3175
  3176, 3177, 3178, ..., 3181, 3182, 3183
  3184, 3185, 3186, ..., 3189, 3190, 3191

  3192, 3193, 3194, ..., 3197, 3198, 3199
  3200, 3201, 3202, ..., 3205, 3206, 3207
  3208, 3209, 3210, ..., 3213, 3214, 3215
  ...,
  3224, 3225, 3226, ..., 3229, 3230, 3231
  3232, 3233, 3234, ..., 3237, 3238, 3239
  3240, 3241, 3242, ..., 3245, 3246, 3247

  3248, 3249, 3250, ..., 3253, 3254, 3255
  3256, 3257, 3258, ..., 3261, 3262, 3263
  3264, 3265, 3266, ..., 3269, 3270, 3271
  ...,
  3280, 3281, 3282, ..., 3285, 3286, 3287
  3288, 3289, 3290, ..., 3293, 3294, 3295
  3296, 3297, 3298, ..., 3301, 3302, 3303

  ...
  ...

  3360, 3361, 3362, ..., 3365, 3366, 3367
  3368, 3369, 3370, ..., 3373, 3374, 3375
  3376, 3377, 3378, ..., 3381, 3382, 3383
  ...,
  3392, 3393, 3394, ..., 3397, 3398, 3399
  3400, 3401, 3402, ..., 3405, 3406, 3407
  3408, 3409, 3410, ..., 3413, 3414, 3415

  3416, 3417, 3418, ..., 3421, 3422, 3423
  3424, 3425, 3426, ..., 3429, 3430, 3431
  3432, 3433, 3434, ..., 3437, 3438, 3439
  ...,
  3448, 3449, 3450, ..., 3453, 3454, 3455
  3456, 3457, 3458, ..., 3461, 3462, 3463
  3464, 3465, 3466, ..., 3469, 3470, 3471

  3472, 3473, 3474, ..., 3477, 3478, 3479
  3480, 3481, 3482, ..., 3485, 3486, 3487
  3488, 3489, 3490, ..., 3493, 3494, 3495
  ...,
  3504, 3505, 3506, ..., 3509, 3510, 3511
  3512, 3513, 3514, ..., 3517, 3518, 3519
  3520, 3521, 3522, ..., 3525, 3526, 3527


  3528, 3529, 3530, ..., 3533, 3534, 3535
  3536, 3537, 3538, ..., 3541, 3542, 3543
  3544, 3545, 3546, ..., 3549, 3550, 3551
  ...,
  3560, 3561, 3562, ..., 3565, 3566, 3567
  3568, 3569, 3570, ..., 3573, 3574, 3575
  3576, 3577, 3578, ..., 3581, 3582, 3583

  3584, 3585, 3586, ..., 3589, 3590, 3591
  3592, 3593, 3594, ..., 3597, 3598, 3599
  3600, 3601, 3602, ..., 3605, 3606, 3607
  ...,
  3616, 3617, 3618, ..., 3621, 3622, 3623
  3624, 3625, 3626, ..., 3629, 3630, 3631
  3632, 3633, 3634, ..., 3637, 3638, 3639

  3640, 3641, 3642, ..., 3645, 3646, 3647
  3648, 3649, 3650, ..., 3653, 3654, 3655
  3656, 3657, 3658, ..., 3661, 3662, 3663
  ...,
  3672, 3673, 3674, ..., 3677, 3678, 3679
  3680, 3681, 3682, ..., 3685, 3686, 3687
  3688, 3689, 3690, ..., 3693, 3694, 3695

  ...
  ...

  3752, 3753, 3754, ..., 3757, 3758, 3759
  3760, 3761, 3762, ..., 3765, 3766, 3767
  3768, 3769, 3770, ..., 3773, 3774, 3775
  ...,
  3784, 3785, 3786, ..., 3789, 3790, 3791
  3792, 3793, 3794, ..., 3797, 3798, 3799
  3800, 3801, 3802, ..., 3805, 3806, 3807

  3808, 3809, 3810, ..., 3813, 3814, 3815
  3816, 3817, 3818, ..., 3821, 3822, 3823
  3824, 3825, 3826, ..., 3829, 3830, 3831
  ...,
  3840, 3841, 3842, ..., 3845, 3846, 3847
  3848, 3849, 3850, ..., 3853, 3854, 3855
  3856, 3857, 3858, ..., 3861, 3862, 3863

  3864, 3865, 3866, ..., 3869, 3870, 3871
  3872, 3873, 3874, ..., 3877, 3878, 3879
  3880, 3881, 3882, ..., 3885, 3886, 3887
  ...,
  3896, 3897, 3898, ..., 3901, 3902, 3903
  3904, 3905, 3906, ..., 3909, 3910, 3911
  3912, 3913, 3914, ..., 3917, 3918, 3919


  3920, 3921, 3922, ..., 3925, 3926, 3927
  3928, 3929, 3930, ..., 3933, 3934, 3935
  3936, 3937, 3938, ..., 3941, 3942, 3943
  ...,
  3952, 3953, 3954, ..., 3957, 3958, 3959
  3960, 3961, 3962, ..., 3965, 3966, 3967
  3968, 3969, 3970, ..., 3973, 3974, 3975

  3976, 3977, 3978, ..., 3981, 3982, 3983
  3984, 3985, 3986, ..., 3989, 3990, 3991
  3992, 3993, 3994, ..., 3997, 3998, 3999
  ...,
  4008, 4009, 4010, ..., 4013, 4014, 4015
  4016, 4017, 4018, ..., 4021, 4022, 4023
  4024, 4025, 4026, ..., 4029, 4030, 4031

  4032, 4033, 4034, ..., 4037, 4038, 4039
  4040, 4041, 4042, ..., 4045, 4046, 4047
  4048, 4049, 4050, ..., 4053, 4054, 4055
  ...,
  4064, 4065, 4066, ..., 4069, 4070, 4071
  4072, 4073, 4074, ..., 4077, 4078, 4079
  4080, 4081, 4082, ..., 4085, 4086, 4087

  ...
  ...

  4144, 4145, 4146, ..., 4149, 4150, 4151
  4152, 4153, 4154, ..., 4157, 4158, 4159
  4160, 4161, 4162, ..., 4165, 4166, 4167
  ...,
  4176, 4177, 4178, ..., 4181, 4182, 4183
  4184, 4185, 4186, ..., 4189, 4190, 4191
  4192, 4193, 4194, ..., 4197, 4198, 4199

  4200, 4201, 4202, ..., 4205, 4206, 4207
  4208, 4209, 4210, ..., 4213, 4214, 4215
  4216, 4217, 4218, ..., 4221, 4222, 4223
  ...,
  4232, 4233, 4234, ..., 4237, 4238, 4239
  4240, 4241, 4242, ..., 4245, 4246, 4247
  4248, 4249, 4250, ..., 4253, 4254, 4255

  4256, 4257, 4258, ..., 4261, 4262, 4263
  4264, 4265, 4266, ..., 4269, 4270, 4271
  4272, 4273, 4274, ..., 4277, 4278, 4279
  ...,
  4288, 4289, 4290, ..., 4293, 4294, 4295
  4296, 4297, 4298, ..., 4301, 4302, 4303
  4304, 4305, 4306, ..., 4309, 4310, 4311


  ...
  ...
  ...


  5096, 5097, 5098, ..., 5101, 5102, 5103
  5104, 5105, 5106, ..., 5109, 5110, 5111
  5112, 5113, 5114, ..., 5117, 5118, 5119
  ...,
  5128, 5129, 5130, ..., 5133, 5134, 5135
  5136, 5137, 5138, ..., 5141, 5142, 5143
  5144, 5145, 5146, ..., 5149, 5150, 5151

  5152, 5153, 5154, ..., 5157, 5158, 5159
  5160, 5161, 5162, ..., 5165, 5166, 5167
  5168, 5169, 5170, ..., 5173, 5174, 5175
  ...,
  5184, 5185, 5186, ..., 5189, 5190, 5191
  5192, 5193, 5194, ..., 5197, 5198, 5199
  5200, 5201, 5202, ..., 5205, 5206, 5207

  5208, 5209, 5210, ..., 5213, 5214, 5215
  5216, 5217, 5218, ..., 5221, 5222, 5223
  5224, 5225, 5226, ..., 5229, 5230, 5231
  ...,
  5240, 5241, 5242, ..., 5245, 5246, 5247
  5248, 5249, 5250, ..., 5253, 5254, 5255
  5256, 5257, 5258, ..., 5261, 5262, 5263

  ...
  ...

  5320, 5321, 5322, ..., 5325, 5326, 5327
  5328, 5329, 5330, ..., 5333, 5334, 5335
  5336, 5337, 5338, ..., 5341, 5342, 5343
  ...,
  5352, 5353, 5354, ..., 5357, 5358, 5359
  5360, 5361, 5362, ..., 5365, 5366, 5367
  5368, 5369, 5370, ..., 5373, 5374, 5375

  5376, 5377, 5378, ..., 5381, 5382, 5383
  5384, 5385, 5386, ..., 5389, 5390, 5391
  5392, 5393, 5394, ..., 5397, 5398, 5399
  ...,
  5408, 5409, 5410, ..., 5413, 5414, 5415
  5416, 5417, 5418, ..., 5421, 5422, 5423
  5424, 5425, 5426, ..., 5429, 5430, 5431

  5432, 5433, 5434, ..., 5437, 5438, 5439
  5440, 5441, 5442, ..., 5445, 5446, 5447
  5448, 5449, 5450, ..., 5453, 5454, 5455
  ...,
  5464, 5465, 5466, ..., 5469, 5470, 5471
  5472, 5473, 5474, ..., 5477, 5478, 5479
  5480, 5481, 5482, ..., 5485, 5486, 5487


  5488, 5489, 5490, ..., 5493, 5494, 5495
  5496, 5497, 5498, ..., 5501, 5502, 5503
  5504, 5505, 5506, ..., 5509, 5510, 5511
  ...,
  5520, 5521, 5522, ..., 5525, 5526, 5527
  5528, 5529, 5530, ..., 5533, 5534, 5535
  5536, 5537, 5538, ..., 5541, 5542, 5543

  5544, 5545, 5546, ..., 5549, 5550, 5551
  5552, 5553, 5554, ..., 5557, 5558, 5559
  5560, 5561, 5562, ..., 5565, 5566, 5567
  ...,
  5576, 5577, 5578, ..., 5581, 5582, 5583
  5584, 5585, 5586, ..., 5589, 5590, 5591
  5592, 5593, 5594, ..., 5597, 5598, 5599

  5600, 5601, 5602, ..., 5605, 5606, 5607
  5608, 5609, 5610, ..., 5613, 5614, 5615
  5616, 5617, 5618, ..., 5621, 5622, 5623
  ...,
  5632, 5633, 5634, ..., 5637, 5638, 5639
  5640, 5641, 5642, ..., 5645, 5646, 5647
  5648, 5649, 5650, ..., 5653, 5654, 5655

  ...
  ...

  5712, 5713, 5714, ..., 5717, 5718, 5719
  5720, 5721, 5722, ..., 5725, 5726, 5727
  5728, 5729, 5730, ..., 5733, 5734, 5735
  ...,
  5744, 5745, 5746, ..., 5749, 5750, 5751
  5752, 5753, 5754, ..., 5757, 5758, 5759
  5760, 5761, 5762, ..., 5765, 5766, 5767

  5768, 5769, 5770, ..., 5773, 5774, 5775
  5776, 5777, 5778, ..., 5781, 5782, 5783
  5784, 5785, 5786, ..., 5789, 5790, 5791
  ...,
  5800, 5801, 5802, ..., 5805, 5806, 5807
  5808, 5809, 5810, ..., 5813, 5814, 5815
  5816, 5817, 5818, ..., 5821, 5822, 5823

  5824, 5825, 5826, ..., 5829, 5830, 5831
  5832, 5833, 5834, ..., 5837, 5838, 5839
  5840, 5841, 5842, ..., 5845, 5846, 5847
  ...,
  5856, 5857, 5858, ..., 5861, 5862, 5863
  5864, 5865, 5866, ..., 5869, 5870, 5871
  5872, 5873, 5874, ..., 5877, 5878, 5879


  5880, 5881, 5882, ..., 5885, 5886, 5887
  5888, 5889, 5890, ..., 5893, 5894, 5895
  5896, 5897, 5898, ..., 5901, 5902, 5903
  ...,
  5912, 5913, 5914, ..., 5917, 5918, 5919
  5920, 5921, 5922, ..., 5925, 5926, 5927
  5928, 5929, 5930, ..., 5933, 5934, 5935

  5936, 5937, 5938, ..., 5941, 5942, 5943
  5944, 5945, 5946, ..., 5949, 5950, 5951
  5952, 5953, 5954, ..., 5957, 5958, 5959
  ...,
  5968, 5969, 5970, ..., 5973, 5974, 5975
  5976, 5977, 5978, ..., 5981, 5982, 5983
  5984, 5985, 5986, ..., 5989, 5990, 5991

  5992, 5993, 5994, ..., 5997, 5998, 5999
  6000, 6001, 6002, ..., 6005, 6006, 6007
  6008, 6009, 6010, ..., 6013, 6014, 6015
  ...,
  6024, 6025, 6026, ..., 6029, 6030, 6031
  6032, 6033, 6034, ..., 6037, 6038, 6039
  6040, 6041, 6042, ..., 6045, 6046, 6047

  ...
  ...

  6104, 6105, 6106, ..., 6109, 6110, 6111
  6112, 6113, 6114, ..., 6117, 6118, 6119
  6120, 6121, 6122, ..., 6125, 6126, 6127
  ...,
  6136, 6137, 6138, ..., 6141, 6142, 6143
  6144, 6145, 6146, ..., 6149, 6150, 6151
  6152, 6153, 6154, ..., 6157, 6158, 6159

  6160, 6161, 6162, ..., 6165, 6166, 6167
  6168, 6169, 6170, ..., 6173, 6174, 6175
  6176, 6177, 6178, ..., 6181, 6182, 6183
  ...,
  6192, 6193, 6194, ..., 6197, 6198, 6199
  6200, 6201, 6202, ..., 6205, 6206, 6207
  6208, 6209, 6210, ..., 6213, 6214, 6215

  6216, 6217, 6218, ..., 6221, 6222, 6223
  6224, 6225, 6226, ..., 6229, 6230, 6231
  6232, 6233, 6234, ..., 6237, 6238, 6239
  ...,
  6248, 6249, 6250, ..., 6253, 6254, 6255
  6256, 6257, 6258, ..., 6261, 6262, 6263
  6264, 6265, 6266, ..., 6269, 6270, 6271
]"#)
}
