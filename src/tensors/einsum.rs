/*!
 * Einstein summation notation
 *
 * A very general purpose sum of products that can represent many
 * different tensor operations with a single notation.
 *
 * See [Einsum].
 */

use crate::numeric::{Numeric, NumericRef};
use crate::tensors::indexing::DynamicShapeIterator;
use crate::tensors::views::{TensorRef, TensorRename, TensorView};
use crate::tensors::{Dimension, Tensor};

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

/**
 * Einstein summation notation
 *
 * A very general purpose sum of products that can represent many
 * different tensor operations with a single notation. In Easy-ML,
 * as tensors are already named, their dimension names are used instead
 * of arbitary characters to refer to dimensions across inputs and
 * the output.
 *
 * Whereas the typical notation used in python libraries
 * is of the form `ab,bc->ac` or `ab->` these would be
 * `Einsum::with_2(&i, &j).to(["a", "c"])` or
 * `Einsum::with_1(&i).to([])` respectively. In scenarios
 * where the existing dimension names in a tensor aren't what you
 * need for the summation notation, there are `named` helper methods
 * to provide an override, so you can perform `ab,bc->ac` with
 * input tensors of different dimension names if you write
 * `Einsum::with_2(&i, &j).named(["a", "b"], ["b", "c"]).to(["a", "c"])`.
 *
 * As with other tensor APIs, the dimension names in a Tensor must be
 * unique, so diagonal summation notation like `aa->` is not supported.
 * Dimensions names can and often will be repeated across input tensors and
 * or the output tensor shape, and each dimension name must have the same length
 * among all of these inputs. APIs will return [InconsistentDimensionLengthError]
 * if a caller passes in inconsistent arguments.
 *
 * See also
 * - [Einsum is All you Need - Einstein Summation in Deep Learning](https://rockt.ai/2018/04/30/einsum)
 * - [Einsum Is All You Need (Video)](https://www.youtube.com/watch?v=pkVwUVEHmfI)
 *
 * To familiarise yourself with translating the Easy-ML specific syntax for
 * Einsum APIs to 'normal' ones, you may also want to look at the
 * [unit tests](https://github.com/Skeletonxf/easy-ml/blob/master/tests/einsum.rs).
 */
#[derive(Clone, Debug, Default)]
pub struct Einsum {
    _private: (),
}

impl Einsum {
    /**
     * An operation with a single input tensor.
     */
    pub fn with_1<T, S, I, const D: usize>(input_1: I) -> Einsum1<T, S, D>
    where
        S: TensorRef<T, D>,
        I: Into<TensorView<T, S, D>>,
    {
        Einsum1 {
            tensor_1: input_1.into(),
        }
    }

    /**
     * An operation with two input tensors.
     */
    pub fn with_2<T, S1, S2, I1, I2, const D1: usize, const D2: usize>(
        input_1: I1,
        input_2: I2,
    ) -> Einsum2<T, S1, S2, D1, D2>
    where
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
        I1: Into<TensorView<T, S1, D1>>,
        I2: Into<TensorView<T, S2, D2>>,
    {
        Einsum2 {
            tensor_1: input_1.into(),
            tensor_2: input_2.into(),
        }
    }

    /**
     * An operation with three input tensors.
     */
    pub fn with_3<T, S1, S2, S3, I1, I2, I3, const D1: usize, const D2: usize, const D3: usize>(
        input_1: I1,
        input_2: I2,
        input_3: I3,
    ) -> Einsum3<T, S1, S2, S3, D1, D2, D3>
    where
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
        S3: TensorRef<T, D3>,
        I1: Into<TensorView<T, S1, D1>>,
        I2: Into<TensorView<T, S2, D2>>,
        I3: Into<TensorView<T, S3, D3>>,
    {
        Einsum3 {
            tensor_1: input_1.into(),
            tensor_2: input_2.into(),
            tensor_3: input_3.into(),
        }
    }

    /**
     * An operation with four input tensors.
     */
    pub fn with_4<
        T,
        S1,
        S2,
        S3,
        S4,
        I1,
        I2,
        I3,
        I4,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const D4: usize,
    >(
        input_1: I1,
        input_2: I2,
        input_3: I3,
        input_4: I4,
    ) -> Einsum4<T, S1, S2, S3, S4, D1, D2, D3, D4>
    where
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
        S3: TensorRef<T, D3>,
        S4: TensorRef<T, D4>,
        I1: Into<TensorView<T, S1, D1>>,
        I2: Into<TensorView<T, S2, D2>>,
        I3: Into<TensorView<T, S3, D3>>,
        I4: Into<TensorView<T, S4, D4>>,
    {
        Einsum4 {
            tensor_1: input_1.into(),
            tensor_2: input_2.into(),
            tensor_3: input_3.into(),
            tensor_4: input_4.into(),
        }
    }
}

/**
 * An error indicating the lengths of dimensions with the same
 * name were inconsistent in the `I` input tensors.
 */
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InconsistentDimensionLengthError<const I: usize> {
    /**
     * The lengths of each matching dimension name in each input
     * in the same order as they were passed to the Einsum APIs.
     *
     * Some inputs may not have this dimension, so will be None.
     */
    pub lengths: [Option<usize>; I],
    /**
     * The dimension name with an inconsistency.
     */
    pub dimension: Dimension,
}

impl<const I: usize> fmt::Display for InconsistentDimensionLengthError<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "inconsistent dimension lengths for dimension '{}': {:?}, lengths must match when repeated in different shapes as the same dimension name",
            self.dimension,
            self.lengths,
        )
    }
}

impl<const I: usize> Error for InconsistentDimensionLengthError<I> {}

#[test]
fn test_inconsistent_dimension_length_error() {
    let error = InconsistentDimensionLengthError {
        lengths: [Some(3), None, Some(2)],
        dimension: "a",
    };
    assert_eq!(
        error.to_string(),
        "inconsistent dimension lengths for dimension 'a': [Some(3), None, Some(2)], lengths must match when repeated in different shapes as the same dimension name",
    )
}

/**
 * A single step in the contractions of an optimised Einsum calculation.
 *
 * The elements in the contraction correspond to indexes for the tensors
 * left in the overall calculation. At the first contraction, there are as many
 * tensor inputs as the number of tensors provided by the caller. For
 * example, a contraction could select the first and third tensors to perform einsum
 * on first, so would be [0, 2]. These tensors are removed from the remaining
 * inputs and we add the results of the einsum operation to the end of the list.
 * If we started with 3 tensors and selected the first and third, we would therefore
 * have two tensors remaining, the second input (now at index 0) and the intermediate
 * tensor we created (now at index 1). Therefore we could have
 * `vec![Contraction::from(vec![0, 2], Contraction::from(vec![0, 1]))]` as our
 * contraction order to split up an einsum calculation into two smaller substeps.
 */
#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct Contraction {
    tensor_indexes: Vec<usize>,
}

// Will come back to using this eventually
#[allow(dead_code)]
impl Contraction {
    /**
     * Creates a Contraction from the input indexes.
     */
    fn from(tensor_indexes: Vec<usize>) -> Contraction {
        Contraction { tensor_indexes }
    }

    /**
     * Returns a reference to the indexes in this contraction.
     */
    fn indexes(&self) -> &[usize] {
        &self.tensor_indexes
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct StepByStepContractionResult {
    input_shapes_left: Vec<Vec<(Dimension, usize)>>,
    contraction_output: Vec<(Dimension, usize)>,
}

/// Given already validated input such that each dimension name repeated over
/// the input_shapes_left and the output_shape share a common dimension length,
/// returns the new list of input_shapes_left and the dimension names for
/// the output of this contraction step.
#[allow(dead_code)]
fn step_by_step_contraction(
    input_shapes_left: &[&[(Dimension, usize)]],
    output_shape: &[(Dimension, usize)],
    contraction: &Contraction,
) -> StepByStepContractionResult {
    // 1. take the (Dimension, usize) shapes out of input_shapes_left matching the Contraction
    // These are the shapes for the tensors we're contracting.
    let contracting: Vec<&[(Dimension, usize)]> = contraction
        .tensor_indexes
        .iter()
        .map(|index| input_shapes_left[*index])
        .collect();

    // 2. make a new list for the other ones not in this contraction (might be empty)
    // These are the shapes for the tensors we'll contract later.
    let not_contracting_yet: Vec<&[(Dimension, usize)]> = input_shapes_left
        .iter()
        .enumerate()
        .filter(|(i, _)| !contraction.tensor_indexes.contains(i))
        .map(|(_, s)| *s)
        .collect();

    // 3. take the union of the dimension names from 1., preserving
    // the order they were originally in the inputs
    // These are the dimensions our contraction will be able to remove via
    // summation if they aren't needed after this step.
    let contracting_dimensions: Vec<(Dimension, usize)> = {
        let mut seen = HashSet::new();
        let mut set = Vec::new();
        for shape in &contracting {
            for d in shape.iter() {
                let new = seen.insert(*d);
                if new {
                    set.push(*d);
                }
            }
        }
        set
    };

    // 4. take the union of the dimension names from the output_shape and 2.,
    // preserving the order they were originally in the inputs
    // These are the dimensions we will still have after this step, due to
    // them being in the final output shape or just required in a later step.
    let retained_dimensions: Vec<(Dimension, usize)> = {
        let mut seen = HashSet::new();
        let mut set = Vec::new();
        for shape in &not_contracting_yet {
            for d in shape.iter() {
                let new = seen.insert(*d);
                if new {
                    set.push(*d);
                }
            }
        }
        for d in output_shape.iter() {
            let new = seen.insert(*d);
            if new {
                set.push(*d);
            }
        }
        set
    };

    // 5. take the dimension names that are in individually in both 4. and 3.
    // These are the dimensions we retain in the contraction at this
    // step.
    let contraction_output: Vec<(Dimension, usize)> = {
        let mut intersection = retained_dimensions.clone();
        intersection.retain(|shape| contracting_dimensions.contains(shape));
        intersection
    };

    // 6. add 2. and new input shape from 5., return to caller to become new input_shapes_left
    // These are the shapes of the tensors left to be contracted
    // in later steps. This will eventually be a single element list
    // matching the output shape when we complete the final step.
    let new_input_shapes_left = {
        let mut vec = Vec::with_capacity(not_contracting_yet.len() + 1);
        for d in not_contracting_yet.iter() {
            vec.push(d.to_vec());
        }
        vec.push(contraction_output.clone());
        vec
    };

    StepByStepContractionResult {
        contraction_output,
        input_shapes_left: new_input_shapes_left,
    }
}

/// Return length of matching dimension in inputs, and error if the length of
/// this output dimension is inconsistent in the input.
fn length_of<const I: usize>(
    output_dimension: Dimension,
    input: &[&[(Dimension, usize)]; I],
) -> Result<usize, InconsistentDimensionLengthError<I>> {
    let lengths = input.map(|shapes| {
        shapes
            .iter()
            .find(|(dimension, _length)| *dimension == output_dimension)
            .map(|(_dimension, length)| *length)
    });

    let first_length = lengths.iter().filter_map(|l| *l).next();
    if let Some(length) = first_length {
        // Check other lengths agree
        if lengths.iter().any(|l| l.is_some() && *l != Some(length)) {
            // Different length matches
            Err(InconsistentDimensionLengthError {
                lengths,
                dimension: output_dimension,
            })
        } else {
            Ok(length)
        }
    } else {
        // No matching lengths, we needed 1 match
        Err(InconsistentDimensionLengthError {
            lengths,
            dimension: output_dimension,
        })
    }
}

#[track_caller]
fn tensor_with_name<T, I, S, const D: usize>(
    dimensions: [Dimension; D],
    tensor: I,
) -> TensorView<T, TensorRename<T, S, D>, D>
where
    I: Into<TensorView<T, S, D>>,
    S: TensorRef<T, D>,
{
    let source: S = tensor.into().source();
    let with_names = TensorRename::from(source, dimensions);
    TensorView::from(with_names)
}

/// Return required output shape given input shape sizes and output shape
/// dimension names. This can fail if a dimension in the requested output
/// shape isn't present in the input, or if the input has contradictory sizes
/// for it.
// We could validate some parts of the input earlier than when we have
// the output dimensions, but validating tensor lengths are consistent for
// each common input dimension name would happen multiple times in the scenario
// of a user using the `named` helper method, so it's a lot easier to use the API
// if we defer validation till the final method call.
fn output_shape_for<const I: usize, const O: usize>(
    input: &[&[(Dimension, usize)]; I],
    output: &[Dimension; O],
) -> Result<[(Dimension, usize); O], InconsistentDimensionLengthError<I>> {
    let mut output_shape = std::array::from_fn(|d| (output[d], 0));
    for x in output_shape.iter_mut() {
        x.1 = length_of(x.0, input)?;
    }
    Ok(output_shape)
}

/// We sum over every dimension included in the input and not the output
///
/// Returns a vec of the summation dimensions along with their validated
/// lengths, and errors if the lengths of any summation dimensions in
/// the input are inconsistent.
fn summation_dimensions<const I: usize, const O: usize>(
    input: &[&[(Dimension, usize)]; I],
    output: &[Dimension; O],
) -> Result<Vec<(Dimension, usize)>, InconsistentDimensionLengthError<I>> {
    let mut total_dimensions = 0;
    for shape in input {
        total_dimensions += shape.len();
    }

    // Worst case is every dimension in each input tensor
    // has unique dimensions
    let mut unique_dimensions = Vec::with_capacity(total_dimensions);

    for shape in input {
        for (dimension, length) in shape.iter() {
            if output.contains(dimension) {
                // If this dimension is requested in the output we will be checking
                // for consistent lengths in `length_of` and this dimension won't be
                // a summation dimension so we can ignore it here.
                continue;
            }
            let existing = unique_dimensions.iter().find(|(d, _)| d == dimension);
            match existing {
                None => unique_dimensions.push((*dimension, *length)),
                Some((_, l)) => {
                    if length != l {
                        // Inconsistent lengths
                        return Err(InconsistentDimensionLengthError {
                            lengths: std::array::from_fn(|i| {
                                input[i]
                                    .iter()
                                    .find(|(d, _)| d == dimension)
                                    .map(|(_, l)| *l)
                            }),
                            dimension,
                        });
                    }
                }
            }
        }
    }

    Ok(unique_dimensions)
}

/// Filters outer indexes to only the matching dimensions
/// for the input shape. Panics if any dimensions in the
/// input shape are missing from the outer slices, but accepts
/// more indexes and dimensions in the outer slices than
/// actually needed for the input shape without any errors.
fn filter_outer_indexes<const D: usize, const O: usize>(
    outer_indexes: &[usize; O],
    outer_shape: &[(Dimension, usize); O],
    input_shape: &[(Dimension, usize); D],
) -> [usize; D] {
    let mut input_indexes = [0; D];
    for d in 0..D {
        let mut found = false;
        let dimension = input_shape[d].0;
        for o in 0..O {
            let possible_dimension = outer_shape[o].0;
            if possible_dimension == dimension {
                input_indexes[d] = outer_indexes[o];
                found = true;
                break;
            }
        }
        if !found {
            panic!(
                "Expected to find an index for dimension {:?} but was not present in {:?} for {:?} while trying to index tensor of shape {:?}",
                dimension,
                outer_indexes,
                outer_shape,
                input_shape,
            );
        }
    }
    input_indexes
}

/// Filters outer indexes and summation indexes to only the
/// matching dimensions for the input shape. Panics if any dimensions
/// in the input shape are missing from the outer and summation slices,
/// but accepts more indexes and dimensions in the outer and summation
/// slices than actually needed for the input shape without any errors.
/// Summation slices must be the same length, we just don't know their
/// length at compile time so can't enforce it in the type system.
fn filter_outer_and_summation_indexes<const D: usize, const O: usize>(
    outer_indexes: &[usize; O],
    outer_shape: &[(Dimension, usize); O],
    summation_indexes: &[usize],
    summation_shape: &[(Dimension, usize)],
    input_shape: &[(Dimension, usize); D],
) -> [usize; D] {
    let mut input_indexes = [0; D];
    for d in 0..D {
        let mut found = false;
        let dimension = input_shape[d].0;
        for o in 0..O {
            let possible_dimension = outer_shape[o].0;
            if possible_dimension == dimension {
                input_indexes[d] = outer_indexes[o];
                found = true;
                break;
            }
        }
        let summation_iter = summation_indexes.iter().zip(summation_shape.iter());
        for (index, (possible_dimension, _length)) in summation_iter {
            if *possible_dimension == dimension {
                input_indexes[d] = *index;
                found = true;
                break;
            }
        }
        if !found {
            panic!(
                "Expected to find an index for dimension {:?} but was not present in {:?} for {:?} or {:?} for {:?} while trying to index tensor of shape {:?}",
                dimension,
                outer_indexes,
                outer_shape,
                summation_indexes,
                summation_shape,
                input_shape,
            );
        }
    }
    input_indexes
}

/**
 * Einstein summation notation operation with a single input tensor.
 */
pub struct Einsum1<T, S1, const D1: usize> {
    tensor_1: TensorView<T, S1, D1>,
}

/**
 * Einstein summation notation operation with two input tensors.
 */
pub struct Einsum2<T, S1, S2, const D1: usize, const D2: usize> {
    tensor_1: TensorView<T, S1, D1>,
    tensor_2: TensorView<T, S2, D2>,
}

/**
 * Einstein summation notation operation with three input tensors
 */
pub struct Einsum3<T, S1, S2, S3, const D1: usize, const D2: usize, const D3: usize> {
    tensor_1: TensorView<T, S1, D1>,
    tensor_2: TensorView<T, S2, D2>,
    tensor_3: TensorView<T, S3, D3>,
}

/**
 * Einstein summation notation operation with four input tensors
 */
pub struct Einsum4<
    T,
    S1,
    S2,
    S3,
    S4,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
> {
    tensor_1: TensorView<T, S1, D1>,
    tensor_2: TensorView<T, S2, D2>,
    tensor_3: TensorView<T, S3, D3>,
    tensor_4: TensorView<T, S4, D4>,
}

impl<T, S1, const D1: usize> Einsum1<T, S1, D1> {
    /**
     * Renames all input tensors to the new names. Their shapes will
     * still be in the same order with the same lengths of data, as
     * per [TensorRename]. As per TensorRename, dimension names for
     * each individual tensor must be unique.
     */
    #[track_caller]
    pub fn named(self, input_1: [Dimension; D1]) -> Einsum1<T, TensorRename<T, S1, D1>, D1>
    where
        S1: TensorRef<T, D1>,
    {
        Einsum1 {
            tensor_1: tensor_with_name(input_1, self.tensor_1),
        }
    }

    pub fn to<const O: usize>(
        self,
        output: [Dimension; O],
    ) -> Result<Tensor<T, O>, InconsistentDimensionLengthError<1>>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        S1: TensorRef<T, D1>,
    {
        let input_1_shape_const = &self.tensor_1.shape();
        let input_1_shape: &[(Dimension, usize)] = input_1_shape_const;
        let input = &[input_1_shape];

        let output_shape = output_shape_for(input, &output)?;
        let mut output_tensor = Tensor::empty(output_shape, T::zero());

        let summation_dimensions = summation_dimensions(input, &output)?;
        let tensor_1_indexing = self.tensor_1.index();

        for (indexes, element) in output_tensor.index_mut().iter_reference_mut().with_index() {
            let mut sum = T::zero();

            if summation_dimensions.is_empty() {
                let product_1 = tensor_1_indexing.get_ref(filter_outer_indexes(
                    &indexes,
                    &output_shape,
                    input_1_shape_const,
                ));
                sum = sum + product_1;
            } else {
                let mut summation_iterator = DynamicShapeIterator::from(&summation_dimensions);
                loop {
                    let next = summation_iterator.next();
                    match next {
                        Some(summation_indexes) => {
                            let product_1 =
                                tensor_1_indexing.get_ref(filter_outer_and_summation_indexes(
                                    &indexes,
                                    &output_shape,
                                    summation_indexes,
                                    &summation_dimensions,
                                    input_1_shape_const,
                                ));
                            sum = sum + product_1;
                        }
                        None => break,
                    }
                }
            }
            *element = sum;
        }

        Ok(output_tensor)
    }
}

impl<T, S1, S2, const D1: usize, const D2: usize> Einsum2<T, S1, S2, D1, D2> {
    /**
     * Renames all input tensors to the new names. Their shapes will
     * still be in the same order with the same lengths of data, as
     * per [TensorRename]. As per TensorRename, dimension names for
     * each individual tensor must be unique.
     */
    #[track_caller]
    pub fn named(
        self,
        input_1: [Dimension; D1],
        input_2: [Dimension; D2],
    ) -> Einsum2<T, TensorRename<T, S1, D1>, TensorRename<T, S2, D2>, D1, D2>
    where
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
    {
        Einsum2 {
            tensor_1: tensor_with_name(input_1, self.tensor_1),
            tensor_2: tensor_with_name(input_2, self.tensor_2),
        }
    }

    pub fn to<const O: usize>(
        self,
        output: [Dimension; O],
    ) -> Result<Tensor<T, O>, InconsistentDimensionLengthError<2>>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
    {
        let input_1_shape_const = &self.tensor_1.shape();
        let input_1_shape: &[(Dimension, usize)] = input_1_shape_const;
        let input_2_shape_const = &self.tensor_2.shape();
        let input_2_shape: &[(Dimension, usize)] = input_2_shape_const;
        let input = &[input_1_shape, input_2_shape];

        let output_shape = output_shape_for(input, &output)?;
        let mut output_tensor = Tensor::empty(output_shape, T::zero());

        let summation_dimensions = summation_dimensions(input, &output)?;
        let tensor_1_indexing = self.tensor_1.index();
        let tensor_2_indexing = self.tensor_2.index();

        for (indexes, element) in output_tensor.index_mut().iter_reference_mut().with_index() {
            let mut sum = T::zero();

            if summation_dimensions.is_empty() {
                let product_1 = tensor_1_indexing.get_ref(filter_outer_indexes(
                    &indexes,
                    &output_shape,
                    input_1_shape_const,
                ));
                let product_2 = tensor_2_indexing.get_ref(filter_outer_indexes(
                    &indexes,
                    &output_shape,
                    input_2_shape_const,
                ));
                sum = sum + (product_1 * product_2);
            } else {
                let mut summation_iterator = DynamicShapeIterator::from(&summation_dimensions);
                loop {
                    let next = summation_iterator.next();
                    match next {
                        Some(summation_indexes) => {
                            let product_1 =
                                tensor_1_indexing.get_ref(filter_outer_and_summation_indexes(
                                    &indexes,
                                    &output_shape,
                                    summation_indexes,
                                    &summation_dimensions,
                                    input_1_shape_const,
                                ));
                            let product_2 =
                                tensor_2_indexing.get_ref(filter_outer_and_summation_indexes(
                                    &indexes,
                                    &output_shape,
                                    summation_indexes,
                                    &summation_dimensions,
                                    input_2_shape_const,
                                ));
                            sum = sum + (product_1 * product_2);
                        }
                        None => break,
                    }
                }
            }

            *element = sum;
        }

        Ok(output_tensor)
    }
}

impl<T, S1, S2, S3, const D1: usize, const D2: usize, const D3: usize>
    Einsum3<T, S1, S2, S3, D1, D2, D3>
{
    /**
     * Renames all input tensors to the new names. Their shapes will
     * still be in the same order with the same lengths of data, as
     * per [TensorRename]. As per TensorRename, dimension names for
     * each individual tensor must be unique.
     */
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn named(
        self,
        input_1: [Dimension; D1],
        input_2: [Dimension; D2],
        input_3: [Dimension; D3],
    ) -> Einsum3<
        T,
        TensorRename<T, S1, D1>,
        TensorRename<T, S2, D2>,
        TensorRename<T, S3, D3>,
        D1,
        D2,
        D3,
    >
    where
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
        S3: TensorRef<T, D3>,
    {
        Einsum3 {
            tensor_1: tensor_with_name(input_1, self.tensor_1),
            tensor_2: tensor_with_name(input_2, self.tensor_2),
            tensor_3: tensor_with_name(input_3, self.tensor_3),
        }
    }

    pub fn to<const O: usize>(
        self,
        output: [Dimension; O],
    ) -> Result<Tensor<T, O>, InconsistentDimensionLengthError<3>>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
        S3: TensorRef<T, D3>,
    {
        let input_1_shape_const = &self.tensor_1.shape();
        let input_1_shape: &[(Dimension, usize)] = input_1_shape_const;
        let input_2_shape_const = &self.tensor_2.shape();
        let input_2_shape: &[(Dimension, usize)] = input_2_shape_const;
        let input_3_shape_const = &self.tensor_3.shape();
        let input_3_shape: &[(Dimension, usize)] = input_3_shape_const;
        let input = &[input_1_shape, input_2_shape, input_3_shape];

        let output_shape = output_shape_for(input, &output)?;
        let mut output_tensor = Tensor::empty(output_shape, T::zero());

        let summation_dimensions = summation_dimensions(input, &output)?;
        let tensor_1_indexing = self.tensor_1.index();
        let tensor_2_indexing = self.tensor_2.index();
        let tensor_3_indexing = self.tensor_3.index();

        for (indexes, element) in output_tensor.index_mut().iter_reference_mut().with_index() {
            let mut sum = T::zero();

            if summation_dimensions.is_empty() {
                let product_1 = tensor_1_indexing.get_ref(filter_outer_indexes(
                    &indexes,
                    &output_shape,
                    input_1_shape_const,
                ));
                let product_2 = tensor_2_indexing.get_ref(filter_outer_indexes(
                    &indexes,
                    &output_shape,
                    input_2_shape_const,
                ));
                let product_3 = tensor_3_indexing.get_ref(filter_outer_indexes(
                    &indexes,
                    &output_shape,
                    input_3_shape_const,
                ));
                sum = sum + (product_1 * product_2 * product_3);
            } else {
                let mut summation_iterator = DynamicShapeIterator::from(&summation_dimensions);
                loop {
                    let next = summation_iterator.next();
                    match next {
                        Some(summation_indexes) => {
                            let product_1 =
                                tensor_1_indexing.get_ref(filter_outer_and_summation_indexes(
                                    &indexes,
                                    &output_shape,
                                    summation_indexes,
                                    &summation_dimensions,
                                    input_1_shape_const,
                                ));
                            let product_2 =
                                tensor_2_indexing.get_ref(filter_outer_and_summation_indexes(
                                    &indexes,
                                    &output_shape,
                                    summation_indexes,
                                    &summation_dimensions,
                                    input_2_shape_const,
                                ));
                            let product_3 =
                                tensor_3_indexing.get_ref(filter_outer_and_summation_indexes(
                                    &indexes,
                                    &output_shape,
                                    summation_indexes,
                                    &summation_dimensions,
                                    input_3_shape_const,
                                ));
                            sum = sum + (product_1 * product_2 * product_3);
                        }
                        None => break,
                    }
                }
            }

            *element = sum;
        }

        Ok(output_tensor)
    }
}

impl<T, S1, S2, S3, S4, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    Einsum4<T, S1, S2, S3, S4, D1, D2, D3, D4>
{
    /**
     * Renames all input tensors to the new names. Their shapes will
     * still be in the same order with the same lengths of data, as
     * per [TensorRename]. As per TensorRename, dimension names for
     * each individual tensor must be unique.
     */
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn named(
        self,
        input_1: [Dimension; D1],
        input_2: [Dimension; D2],
        input_3: [Dimension; D3],
        input_4: [Dimension; D4],
    ) -> Einsum4<
        T,
        TensorRename<T, S1, D1>,
        TensorRename<T, S2, D2>,
        TensorRename<T, S3, D3>,
        TensorRename<T, S4, D4>,
        D1,
        D2,
        D3,
        D4,
    >
    where
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
        S3: TensorRef<T, D3>,
        S4: TensorRef<T, D4>,
    {
        Einsum4 {
            tensor_1: tensor_with_name(input_1, self.tensor_1),
            tensor_2: tensor_with_name(input_2, self.tensor_2),
            tensor_3: tensor_with_name(input_3, self.tensor_3),
            tensor_4: tensor_with_name(input_4, self.tensor_4),
        }
    }

    pub fn to<const O: usize>(
        self,
        output: [Dimension; O],
    ) -> Result<Tensor<T, O>, InconsistentDimensionLengthError<4>>
    where
        T: Numeric,
        for<'a> &'a T: NumericRef<T>,
        S1: TensorRef<T, D1>,
        S2: TensorRef<T, D2>,
        S3: TensorRef<T, D3>,
        S4: TensorRef<T, D4>,
    {
        let input_1_shape_const = &self.tensor_1.shape();
        let input_1_shape: &[(Dimension, usize)] = input_1_shape_const;
        let input_2_shape_const = &self.tensor_2.shape();
        let input_2_shape: &[(Dimension, usize)] = input_2_shape_const;
        let input_3_shape_const = &self.tensor_3.shape();
        let input_3_shape: &[(Dimension, usize)] = input_3_shape_const;
        let input_4_shape_const = &self.tensor_4.shape();
        let input_4_shape: &[(Dimension, usize)] = input_4_shape_const;
        let input = &[input_1_shape, input_2_shape, input_3_shape, input_4_shape];

        let output_shape = output_shape_for(input, &output)?;
        let mut output_tensor = Tensor::empty(output_shape, T::zero());

        let summation_dimensions = summation_dimensions(input, &output)?;
        let tensor_1_indexing = self.tensor_1.index();
        let tensor_2_indexing = self.tensor_2.index();
        let tensor_3_indexing = self.tensor_3.index();
        let tensor_4_indexing = self.tensor_4.index();

        for (indexes, element) in output_tensor.index_mut().iter_reference_mut().with_index() {
            let mut sum = T::zero();

            if summation_dimensions.is_empty() {
                let product_1 = tensor_1_indexing.get_ref(filter_outer_indexes(
                    &indexes,
                    &output_shape,
                    input_1_shape_const,
                ));
                let product_2 = tensor_2_indexing.get_ref(filter_outer_indexes(
                    &indexes,
                    &output_shape,
                    input_2_shape_const,
                ));
                let product_3 = tensor_3_indexing.get_ref(filter_outer_indexes(
                    &indexes,
                    &output_shape,
                    input_3_shape_const,
                ));
                let product_4 = tensor_4_indexing.get_ref(filter_outer_indexes(
                    &indexes,
                    &output_shape,
                    input_4_shape_const,
                ));
                sum = sum + (product_1 * product_2 * product_3 * product_4);
            } else {
                let mut summation_iterator = DynamicShapeIterator::from(&summation_dimensions);
                loop {
                    let next = summation_iterator.next();
                    match next {
                        Some(summation_indexes) => {
                            let product_1 =
                                tensor_1_indexing.get_ref(filter_outer_and_summation_indexes(
                                    &indexes,
                                    &output_shape,
                                    summation_indexes,
                                    &summation_dimensions,
                                    input_1_shape_const,
                                ));
                            let product_2 =
                                tensor_2_indexing.get_ref(filter_outer_and_summation_indexes(
                                    &indexes,
                                    &output_shape,
                                    summation_indexes,
                                    &summation_dimensions,
                                    input_2_shape_const,
                                ));
                            let product_3 =
                                tensor_3_indexing.get_ref(filter_outer_and_summation_indexes(
                                    &indexes,
                                    &output_shape,
                                    summation_indexes,
                                    &summation_dimensions,
                                    input_3_shape_const,
                                ));
                            let product_4 =
                                tensor_4_indexing.get_ref(filter_outer_and_summation_indexes(
                                    &indexes,
                                    &output_shape,
                                    summation_indexes,
                                    &summation_dimensions,
                                    input_4_shape_const,
                                ));
                            sum = sum + (product_1 * product_2 * product_3 * product_4);
                        }
                        None => break,
                    }
                }
            }

            *element = sum;
        }

        Ok(output_tensor)
    }
}

#[test]
fn step_by_step_contraction_tests() {
    // Simple case where we just contract 2 tensors and consume all the input
    assert_eq!(
        step_by_step_contraction(
            &[&[("x", 2), ("y", 3)], &[("y", 3), ("z", 4)]],
            &[("x", 2), ("z", 4)],
            &Contraction {
                tensor_indexes: vec![0, 1]
            },
        ),
        StepByStepContractionResult {
            input_shapes_left: vec![vec![("x", 2), ("z", 4)]],
            contraction_output: vec![("x", 2), ("z", 4)],
        }
    );
    // Case where we contract out `b` and `d` and leave just two tensors
    // with `a` and `c` terms to contract next.
    #[rustfmt::skip]
    assert_eq!(
        step_by_step_contraction(
            &[
                &[("a", 2), ("b", 3), ("d", 5)],
                &[("a", 2), ("c", 4)],
                &[("b", 3), ("d", 5), ("c", 4)],
            ],
            &[("a", 2), ("c", 4)],
            &Contraction {
                tensor_indexes: vec![0, 2]
            },
        ),
        StepByStepContractionResult {
            input_shapes_left: vec![
                vec![("a", 2), ("c", 4)],
                vec![("a", 2), ("c", 4)],
            ],
            contraction_output: vec![("a", 2), ("c", 4)],
        }
    );
    // Less optimised route where we have to leave `b` and `d` terms in
    // because the last input still needs them, and we can't contract out
    // `a` because we requested it in the output.
    assert_eq!(
        step_by_step_contraction(
            &[
                &[("a", 2), ("b", 3), ("d", 5)],
                &[("a", 2), ("c", 4)],
                &[("b", 3), ("d", 5), ("c", 4)],
            ],
            &[("a", 2), ("c", 4)],
            &Contraction {
                tensor_indexes: vec![0, 1]
            },
        ),
        StepByStepContractionResult {
            input_shapes_left: vec![
                vec![("b", 3), ("d", 5), ("c", 4)],
                vec![("b", 3), ("d", 5), ("c", 4), ("a", 2)],
            ],
            contraction_output: vec![("b", 3), ("d", 5), ("c", 4), ("a", 2)],
        }
    );
    // Slightly different route where we can contract out `a` because
    // we didn't request it in the output.
    assert_eq!(
        step_by_step_contraction(
            &[
                &[("a", 2), ("b", 3), ("d", 5)],
                &[("a", 2), ("c", 4)],
                &[("b", 3), ("d", 5), ("c", 4)],
            ],
            &[("c", 4)],
            &Contraction {
                tensor_indexes: vec![0, 1]
            },
        ),
        StepByStepContractionResult {
            input_shapes_left: vec![
                vec![("b", 3), ("d", 5), ("c", 4)],
                vec![("b", 3), ("d", 5), ("c", 4)],
            ],
            contraction_output: vec![("b", 3), ("d", 5), ("c", 4)],
        }
    );
}

// TODO: Letting caller pass in the desired contraction order
// should largely build on top of naive Einsum implementation, but we
// are going to need quite a few more APIs to generalise over different
// dimensionalities of tensors first since we have to erase dimension length
// and dimension arguments somehow.
// fn by_contraction_order<const O: usize>(
//     self,
//     output: [Dimension; O],
//     contraction_order: &[Contraction],
// ) -> Result<Tensor<T, O>, InconsistentDimensionLengthError<3>>
// where
//     T: Numeric,
//     for<'a> &'a T: NumericRef<T>,
//     S1: TensorRef<T, D1>,
//     S2: TensorRef<T, D2>,
//     S3: TensorRef<T, D3>,
// {
//     let input_1_shape_const = &self.tensor_1.shape();
//     let input_1_shape: &[(Dimension, usize)] = input_1_shape_const;
//     let input_2_shape_const = &self.tensor_2.shape();
//     let input_2_shape: &[(Dimension, usize)] = input_2_shape_const;
//     let input_3_shape_const = &self.tensor_3.shape();
//     let input_3_shape: &[(Dimension, usize)] = input_3_shape_const;
//     let input = &[input_1_shape, input_2_shape, input_3_shape];
//
//     let output_shape = output_shape_for(input, &output)?;
//     let mut output_tensor = Tensor::empty(output_shape, T::zero());
//
//     let summation_dimensions = summation_dimensions(input, &output)?;
//
//     let mut input: Vec<Vec<(Dimension, usize)>> = input.iter().map(|i| i.to_vec()).collect();
//
//     for contraction in contraction_order.iter() {
//         let step = step_by_step_contraction(
//             &input.iter().map(|i| i.as_slice()).collect::<Vec<&[(Dimension, usize)]>>(),
//             &output_shape,
//             &contraction,
//         );
//         input = step.input_shapes_left;
//         let einsum_step = step.contraction_output;
//         // Need to store the unprocessed inputs in a list somehow which
//         // is going to require first erasing or at least enumerating over
//         // their dimensionality.
//         match einsum_step.len() {
//             0 => unimplemented!(),
//             1 => Einsum::with_1(...),
//             2 => Einsum::with_2(...),
//             3 => Einsum::with_3(...),
//             _ => panic!("Unsupported contraction step, output was larger than supported")
//         }
//     }
//
//     unimplemented!()
// }

// TODO: Once Tensor implementation is working, should be able to actually generalise
// to work on RecordTensor inputs too, they can be passed in up to the .to() step already.
// Final step needs to be aware of some kind of NumericLike type that knows how to lift
// and lower from additional context to a Numeric type for addition and multiplication.
// In some future work can introduce a generic associated type for TensorRef that
// 'knows' what the desired container output is for tensor operations like these to collect
// the results back into, so result type becomes Result<S1::Output<T, O>, InconsistentDimensionLength>
// and somehow we enforce S1::Output == S2::Output????
