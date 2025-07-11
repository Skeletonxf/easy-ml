/*!
 * Utilities to manipulate Dimensions.
 *
 * Contains a number of utility functions for manipulating the dimensions of tensors.
 *
 * # Terminology
 *
 * Tensors in Easy ML have a **shape** of type `[(Dimension, usize); D]`, where D is a compile time
 * constant. This type defines a list of dimension names and the lengths along each dimension name.
 * A tensor with some shape will have data ranging from 0 to the length - 1 along each
 * dimension name.
 * Often we want to call methods which only take dimension names in some order, so those
 * **dimension names** have a type of `[Dimension; D]`. We also may want to index into a Tensor,
 * which is done by providing the **index**es only, with a type of `[usize; D]`.
 *
 * Dimensions and dimension names in Easy ML APIs are treated like lists, the order of
 * each dimension does make a difference for equality definitions, mathematical operations,
 * and can be a factor for the order of iteration and indexing. However, many high level APIs
 * that are not directly involved with order or indexing require only a dimension name and these
 * are usually less concerned with the order of the dimensions.
 */

use crate::tensors::Dimension;

/**
 * Returns the product of the dimension lengths in the provided shape.
 *
 * This is equal to the number of elements that will be stored for these dimensions.
 * A 0 dimensional tensor stores exactly 1 element, a 1 dimensional tensor stores N elements,
 * a 2 dimensional tensor stores NxM elements and so on.
 */
pub fn elements<const D: usize>(shape: &[(Dimension, usize); D]) -> usize {
    shape.iter().map(|d| d.1).product()
}

/**
 * Finds the position of the dimension name in the shape.
 *
 * `None` is returned if the dimension name is not in the shape.
 */
pub fn position_of<const D: usize>(
    shape: &[(Dimension, usize); D],
    dimension: Dimension,
) -> Option<usize> {
    shape.iter().position(|(d, _)| d == &dimension)
}

/**
 * Checks if the dimension name is in the shape.
 */
pub fn contains<const D: usize>(shape: &[(Dimension, usize); D], dimension: Dimension) -> bool {
    shape.iter().any(|(d, _)| d == &dimension)
}

/**
 * Returns the length of the dimension name provided, if one is present in the shape.
 */
pub fn length_of<const D: usize>(
    shape: &[(Dimension, usize); D],
    dimension: Dimension,
) -> Option<usize> {
    shape
        .iter()
        .find(|(d, _)| *d == dimension)
        .map(|(_, length)| *length)
}

/**
 * Returns the last index of the dimension name provided, if one is present in the shape.
 *
 * This is always 1 less than the length, the 'index' in this sense is based on what the
 * shape is, not any implementation index. If for some reason a shape with a 0
 * length was given, the last index will saturate at 0.
 */
pub fn last_index_of<const D: usize>(
    shape: &[(Dimension, usize); D],
    dimension: Dimension,
) -> Option<usize> {
    length_of(shape, dimension).map(|length| length.saturating_sub(1))
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct DimensionMappings<const D: usize> {
    source_to_requested: [usize; D],
    requested_to_source: [usize; D],
}

impl<const D: usize> DimensionMappings<D> {
    pub(crate) fn no_op_mapping() -> DimensionMappings<D> {
        DimensionMappings {
            source_to_requested: std::array::from_fn(|d| d),
            requested_to_source: std::array::from_fn(|d| d),
        }
    }

    // Computes both mappings from from a shape in source order and a matching set of
    // dimensions in an arbitary order.
    // If the source order is x,y,z but the requested order is z,x,y then the mapping
    // from source to requested is [1,2,0] (x becomes second, y becomes last, z becomes first) and
    // from requested to source is [2,0,1] (z becones last, x becomes first, y becomes second).
    pub(crate) fn new(
        source: &[(Dimension, usize); D],
        requested: &[Dimension; D],
    ) -> Option<DimensionMappings<D>> {
        let mut source_to_requested = [0; D];
        let mut requested_to_source = [0; D];
        for d in 0..D {
            let dimension = source[d].0;
            // happy path, requested dimension is in the same order as in source order
            if requested[d] == dimension {
                source_to_requested[d] = d;
                requested_to_source[d] = d;
            } else {
                // If dimensions are in a different order, find the dimension with the
                // matching dimension name for both mappings at this position in the order.
                // Since both lists are the same length and we know our source order won't contain
                // duplicates this also ensures the two lists have exactly the same set of names
                // as otherwise one of these `find`s will fail.
                let (n_in_requested, _) = requested
                    .iter()
                    .enumerate()
                    .find(|(_, d)| **d == dimension)?;
                source_to_requested[d] = n_in_requested;
                let dimension = requested[d];
                let (n_in_source, _) = source
                    .iter()
                    .enumerate()
                    .find(|(_, (d, _))| *d == dimension)?;
                requested_to_source[d] = n_in_source;
            };
        }
        Some(DimensionMappings {
            source_to_requested,
            requested_to_source,
        })
    }

    /// Reorders some indexes according to the dimension mapping to return the
    /// indexes in the source order
    #[inline]
    pub(crate) fn map_dimensions_to_source(&self, indexes: &[usize; D]) -> [usize; D] {
        // Our input is in requested order and we return indexes in the source order, so for each
        // dimension to return (in source order) we're looking up which index from the input to
        // use, just like for map_linear_data_layout_to_transposed.
        std::array::from_fn(|d| indexes[self.source_to_requested[d]])
    }

    /// Reorders some shape according to the dimension mapping to return the
    /// shape in the requested order
    #[inline]
    pub(crate) fn map_shape_to_requested(
        &self,
        source: &[(Dimension, usize); D],
    ) -> [(Dimension, usize); D] {
        // For each d we're returning, we're giving what the requested dth dimension is
        // in the source, so we use requested_to_source for mapping. This is different to indexing
        // or map_linear_data_layout_to_transposed because our output here is in requested order,
        // not our input.
        std::array::from_fn(|d| source[self.requested_to_source[d]])
    }

    #[inline]
    pub(crate) fn map_linear_data_layout_to_transposed(
        &self,
        order: &[Dimension; D],
    ) -> [Dimension; D] {
        // In most simple cases the transformation for source -> requested is the same as
        // requested -> source so the data layout maps the same way as the shape. However,
        // in most complex 3D or higher cases, we can transpose a tensor in such a way that
        // the data_layout does not map the same way. To conceptualise this, consider we first
        // just used a TensorAccess to swap the indexes from ["batch", "row", "column"] to
        // ["row", "column", "batch"]. This is a mapping of [0->2, 1->0, 2->1] (or if we only
        // want to write where each dimension ends up, we can call this a mapping of [2, 0, 1]).
        // TensorAccess doesn't change what each dimension refers to in memory, so the data_layout
        // stays as ["batch", "row", "column"]. We can then wrap our TensorAccess in a TensorRename
        // to emulate the behavior of a TensorTranspose. We rename ["row", "column", "batch"]
        // to ["batch", "row", "column"] again, retaining our swapped dimensions as they are.
        // Now row in our data_layout is called batch, column becomes row, and batch becomes
        // column. Hence our data layout becames ["column", "batch", "row"]. If we compare our
        // data layout to what we started with (["batch", "row", "column"]) we see that we have
        // mapped it as [1, 2, 0], not [2, 0, 1]. As a more general explanation, the data layout
        // we return here is mapping from what the data layout of the source is to the order
        // we're now requesting in, so we use the reverse mapping to what we use for the
        // view_shape. This ensures the data layout we return is correct, which can always be
        // sanity checked by constructing a TensorAccess from what we return and verifying that
        // the data is restored to memory order (assuming the original source was in the same
        // endianess as Tensor).
        std::array::from_fn(|d| order[self.source_to_requested[d]])
    }
}

/**
 * Returns true if the dimensions are all the same length. For 0 or 1 dimensions trivially returns
 * true. For 2 dimensions, this corresponds to a square matrix, and for 3 dimensions, a cube shaped
 * tensor, and so on.
 */
pub fn is_square<const D: usize>(shape: &[(Dimension, usize); D]) -> bool {
    if D > 1 {
        let first = shape[0].1;
        #[allow(clippy::needless_range_loop)]
        for d in 1..D {
            if shape[d].1 != first {
                return false;
            }
        }
        true
    } else {
        true
    }
}

/**
 * Returns just the dimension names of the shape, in the same order.
 */
pub fn names_of<const D: usize>(shape: &[(Dimension, usize); D]) -> [Dimension; D] {
    shape.map(|(dimension, _length)| dimension)
}

pub(crate) fn has_duplicates(shape: &[(Dimension, usize)]) -> bool {
    for i in 1..shape.len() {
        let name = shape[i - 1].0;
        if shape[i..].iter().any(|d| d.0 == name) {
            return true;
        }
    }
    false
}

pub(crate) fn has_duplicates_names(dimensions: &[Dimension]) -> bool {
    for i in 1..dimensions.len() {
        let name = dimensions[i - 1];
        if dimensions[i..].contains(&name) {
            return true;
        }
    }
    false
}

pub(crate) fn has_duplicates_extra_names(dimensions: &[(usize, Dimension)]) -> bool {
    for i in 1..dimensions.len() {
        let name = dimensions[i - 1].1;
        if dimensions[i..].iter().any(|&d| d.1 == name) {
            return true;
        }
    }
    false
}

#[test]
fn test_dimension_mappings() {
    let shape = [("x", 0), ("y", 0), ("z", 0)];

    let mapping = DimensionMappings::new(&shape, &["x", "y", "z"]).unwrap();
    assert_eq!(
        DimensionMappings {
            source_to_requested: [0, 1, 2],
            requested_to_source: [0, 1, 2],
        },
        mapping
    );
    assert_eq!(
        [("x", 0), ("y", 0), ("z", 0)],
        mapping.map_shape_to_requested(&shape),
    );

    let mapping = DimensionMappings::new(&shape, &["z", "y", "x"]).unwrap();
    assert_eq!(
        DimensionMappings {
            source_to_requested: [2, 1, 0],
            requested_to_source: [2, 1, 0],
        },
        mapping
    );
    assert_eq!(
        [("z", 0), ("y", 0), ("x", 0)],
        mapping.map_shape_to_requested(&shape),
    );

    let mapping = DimensionMappings::new(&shape, &["z", "x", "y"]).unwrap();
    assert_eq!(
        DimensionMappings {
            source_to_requested: [1, 2, 0],
            requested_to_source: [2, 0, 1],
        },
        mapping
    );
    assert_eq!(
        [("z", 0), ("x", 0), ("y", 0)],
        mapping.map_shape_to_requested(&shape),
    );

    let mapping = DimensionMappings::new(&shape, &["x", "z", "y"]).unwrap();
    assert_eq!(
        DimensionMappings {
            source_to_requested: [0, 2, 1],
            requested_to_source: [0, 2, 1],
        },
        mapping
    );
    assert_eq!(
        [("x", 0), ("z", 0), ("y", 0)],
        mapping.map_shape_to_requested(&shape),
    );

    let mapping = DimensionMappings::new(&shape, &["y", "z", "x"]).unwrap();
    assert_eq!(
        DimensionMappings {
            source_to_requested: [2, 0, 1],
            requested_to_source: [1, 2, 0],
        },
        mapping
    );
    assert_eq!(
        [("y", 0), ("z", 0), ("x", 0)],
        mapping.map_shape_to_requested(&shape),
    );
}

#[test]
fn test_is_square() {
    assert_eq!(true, is_square(&[]));
    assert_eq!(true, is_square(&[("x", 1)]));
    assert_eq!(true, is_square(&[("x", 1), ("y", 1)]));
    assert_eq!(true, is_square(&[("x", 4), ("y", 4)]));
    assert_eq!(false, is_square(&[("x", 4), ("y", 3)]));
    assert_eq!(true, is_square(&[("x", 3), ("y", 3), ("z", 3)]));
    assert_eq!(false, is_square(&[("x", 3), ("y", 4), ("z", 3)]));
}

#[test]
fn test_duplicate_names() {
    assert_eq!(has_duplicates_names(&["a", "b", "b", "c"]), true);
    assert_eq!(has_duplicates_names(&["a", "b", "c", "d"]), false);
    assert_eq!(has_duplicates_names(&["a", "b", "a", "c"]), true);
    assert_eq!(has_duplicates_names(&["a", "a", "a", "a"]), true);
    assert_eq!(has_duplicates_names(&["a", "b", "c", "c"]), true);
}
