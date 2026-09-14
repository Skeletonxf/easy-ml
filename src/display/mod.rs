use crate::matrices::views::MatrixRef;

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
