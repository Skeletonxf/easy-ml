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
 * dimension name. Many APIs simply refer to this type as the **dimensions**.
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
 * Returns the product of the provided dimension lengths
 *
 * This is equal to the number of elements that will be stored for these dimensions.
 * A 0 dimensional tensor stores exactly 1 element, a 1 dimensional tensor stores N elements,
 * a 2 dimensional tensor stores NxM elements and so on.
 */
pub fn elements<const D: usize>(dimensions: &[(Dimension, usize); D]) -> usize {
    dimensions.iter().map(|d| d.1).product()
}

/**
 * Finds the position of the dimension name in the set of dimensions.
 *
 * `None` is returned if the dimension name is not in the set.
 */
pub fn position_of<const D: usize>(
    dimensions: &[(Dimension, usize); D],
    dimension: Dimension,
) -> Option<usize> {
    dimensions.iter().position(|(d, _)| d == &dimension)
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

    // Computes both mappings from from a set of dimensions in source order and a matching set of
    // dimensions in an arbitary order.
    // If the source order is x,y,z but the requested order is z,x,y then the mapping
    // from source to requested is [1,2,0] (x becomes second, y becomes last, z becomes first) and
    // from requested to source is [2,0,1] (z becones kast, x becomes first, y becomes second).
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
        Some(
            DimensionMappings {
                source_to_requested,
                requested_to_source,
            }
        )
    }

    // Reorders some indexes according to the dimension mapping to return the
    // indexes in the source order
    #[inline]
    pub(crate) fn map_dimensions_to_source(
        &self,
        indexes: &[usize; D],
    ) -> [usize; D] {
        // Our input is in requested order, so we need to lookup the order in the source
        // to return the correct indexes.
        std::array::from_fn(|d| indexes[self.requested_to_source[d]])
    }

    // Reorders some shape according to the dimension mapping to return the
    // shape in the requested order
    #[inline]
    pub(crate) fn map_shape_to_requested(
        &self,
        source: &[(Dimension, usize); D],
    ) -> [(Dimension, usize); D] {
        // For each d we're returning, we're giving what the requested dth dimension is
        // in the source, so we reorder with requested_to_source even though we're mapping
        // an input in source order to requested order.
        std::array::from_fn(|d| source[self.requested_to_source[d]])
    }

    // Reorders some source linear data layout according to the dimension mapping
    // to return the new linear data layout order for what
    // the mapped shape will be.
    #[inline]
    pub(crate) fn map_linear_data_layout_to_requested(
        &self,
        source: &[usize; D],
    ) -> [usize; D] {
        // This is identical to map_shape_to_requested because the swap of dimensions and
        // corresponding swap on the view shape means the data layout order swaps the same way.
        std::array::from_fn(|d| source[self.requested_to_source[d]])
    }
}

/**
 * Returns true if the dimensions are all the same length. For 0 or 1 dimensions trivially returns
 * true. For 2 dimensions, this corresponds to a square matrix, and for 3 dimensions, a cube shaped
 * tensor, and so on.
 */
pub fn is_square<const D: usize>(dimensions: &[(Dimension, usize); D]) -> bool {
    if D > 1 {
        let first = dimensions[0].1;
        for d in 1..D {
            if dimensions[d].1 != first {
                return false;
            }
        }
        true
    } else {
        true
    }
}

/**
 * Returns just the dimension names of the dimensions, in the same order.
 */
pub fn names_of<const D: usize>(dimensions: &[(Dimension, usize); D]) -> [Dimension; D] {
    dimensions.map(|(dimension, _length)| dimension)
}

pub(crate) fn has_duplicates(dimensions: &[(Dimension, usize)]) -> bool {
    for i in 1..dimensions.len() {
        let name = dimensions[i - 1].0;
        if dimensions[i..].iter().any(|d| d.0 == name) {
            return true;
        }
    }
    false
}

pub(crate) fn has_duplicates_names(dimensions: &[Dimension]) -> bool {
    for i in 1..dimensions.len() {
        let name = dimensions[i - 1];
        if dimensions[i..].iter().any(|&d| d == name) {
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
