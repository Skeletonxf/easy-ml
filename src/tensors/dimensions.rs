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

// Computes a mapping from a set of dimensions in source order to a matching set of
// dimensions in an arbitary order.
// Returns a list where each dimension in the source order is mapped to the requested order,
// such that if the source order is x,y,z but the requested order is z,y,x then the mapping
// is [2,1,0] as this maps the first dimension x to the third dimension x, the second dimension y
// to the second dimension y, and the third dimension z to to the first dimension z.
pub(crate) fn dimension_mapping<const D: usize>(
    source: &[(Dimension, usize); D],
    requested: &[Dimension; D],
) -> Option<[usize; D]> {
    let mut mapping = [0; D];
    for d in 0..D {
        let dimension = source[d].0;
        // happy path, requested dimension is in the same order as in source order
        let order = if requested[d] == dimension {
            d
        } else {
            // If dimensions are in a different order, find the requested dimension with the
            // matching dimension name.
            // Since both lists are the same length and we know our source order won't contain
            // duplicates this also ensures the two lists have exactly the same set of names
            // as otherwise one of these `find`s will fail.
            let (n, _) = requested
                .iter()
                .enumerate()
                .find(|(_, d)| **d == dimension)?;
            n
        };
        mapping[d] = order;
    }
    Some(mapping)
}

// pub(crate) fn same_dimensions<const D: usize>(
//     source: &[(Dimension, usize); D],
//     requested: &[Dimension; D],
// ) -> bool {
//     dimension_mapping(source, requested).is_some()
// }

// Reorders some indexes according to the dimension_mapping to return indexes in source order from
// input indexes in the arbitary order
#[inline]
pub(crate) fn map_dimensions<const D: usize>(
    dimension_mapping: &[usize; D],
    indexes: &[usize; D],
) -> [usize; D] {
    let mut lookup = [0; D];
    for d in 0..D {
        lookup[d] = indexes[dimension_mapping[d]];
    }
    lookup
}

pub(crate) fn dimension_mapping_shape<const D: usize>(
    source: &[(Dimension, usize); D],
    dimension_mapping: &[usize; D],
) -> [(Dimension, usize); D] {
    #[allow(clippy::clone_on_copy)]
    let mut shape = source.clone();
    for d in 0..D {
        // The ith dimension of the mapped shape has a length of the jth dimension length where
        // dimension_mapping maps from the ith dimension (of some arbitary dimension order) to
        // the jth dimension (of the order in source)
        shape[d] = source[dimension_mapping[d]];
    }
    shape
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

#[test]
fn test_dimension_mapping() {
    let mapping = dimension_mapping(&[("x", 0), ("y", 0), ("z", 0)], &["x", "y", "z"]);
    assert_eq!([0, 1, 2], mapping.unwrap());
    let mapping = dimension_mapping(&[("x", 0), ("y", 0), ("z", 0)], &["z", "y", "x"]);
    assert_eq!([2, 1, 0], mapping.unwrap());
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
