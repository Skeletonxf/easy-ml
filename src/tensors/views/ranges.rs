use crate::tensors::views::TensorRef;
use crate::tensors::{Dimension, InvalidShapeError, InvalidDimensionsError};
use std::marker::PhantomData;

use crate::matrices::views::IndexRange;

// two ways to mask, together they allow arbitary mask shapes:
// mask by x to y EXCLUDED, keeping the rest
// mask by x to y INCLUDED, dropping the rest

/**
 * A range over a tensor in D dimensions, hiding the values **outside** the range from view.
 *
 * The entire source is still owned by the TensorRange however, so this does not permit
 * creating multiple mutable ranges into a single tensor even if they wouldn't overlap.
 */
#[derive(Clone, Debug)]
pub struct TensorRange<T, S, const D: usize> {
    source: S,
    range: [IndexRange; D],
    _type: PhantomData<T>,
}

/**
 * A mask over a tensor in D dimensions, hiding the values **inside** the range from view.
 *
 * The entire source is still owned by the TensorMask however, so this does not permit
 * creating multiple mutable masks into a single tensor even if they wouldn't overlap.
 */
#[derive(Clone, Debug)]
pub struct TensorMask<T, S, const D: usize> {
    source: S,
    mask: [IndexRange; D],
    _type: PhantomData<T>,
}

/**
 * An error in creating a [TensorRange](TensorRange) or a [TensorMask](TensorMask).
 */
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IndexRangeValidationError<const D: usize, const P: usize> {
    /**
     * The shape that resulting Tensor would have would not be valid.
     */
    InvalidShape(InvalidShapeError<D>),
    /**
     * Multiple of the same dimension name were provided, but we can only take one mask or range
     * for each dimension at a time.
     */
    InvalidDimensions(InvalidDimensionsError<D, P>),
}

/**
 * An error in creating a [TensorRange](TensorRange) or a [TensorMask](TensorMask) using
 * strict validation.
 */
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StrictIndexRangeValidationError<const D: usize, const P: usize> {
    /**
     * In at least one dimension, the mask or range provided exceeds the bounds of the shape
     * of the Tensor it was to be used on. This is not necessarily an issue as the mask or
     * range could be clipped to the bounds of the Tensor's shape, but a constructor which
     * rejects out of bounds input was used.
     */
    OutsideShape {
        shape: [(Dimension, usize); D],
        index_range: [Option<IndexRange>; D],
    },
    Error(IndexRangeValidationError<D, P>),
}

impl<T, S, const D: usize> TensorRange<T, S, D>
where
    S: TensorRef<T, D>,
{
    // TODO: Alternate named constructors that take M dimension names and index ranges and call into
    // from/from_strict

    pub fn from<R, const P: usize>(
        source: S,
        ranges: [(Dimension, R); P]
    ) -> Result<TensorRange<T, S, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>
    {
        let shape = source.view_shape();
        let names_in_tensor = shape.map(|(d, _)| d);
        let names_in_ranges = ranges.map(|(d, _)| d);
        let dimensions = InvalidDimensionsError {
            provided: names_in_ranges,
            valid: names_in_tensor,
        };
        if dimensions.has_duplicates() {
            return Err(IndexRangeValidationError::InvalidDimensions(dimensions));
        }
        // TODO: Check provided is subset of valid
        // TODO: Call into from_all
        unimplemented!()
    }

    /**
     * Constructs a TensorRange from a tensor and a range for each dimension in the tensor
     * (provided in the same order as the tensor's shape).
     *
     * Returns the Err variant if any dimension would have a length of 0.
     */
    pub fn from_all<R>(
        source: S,
        range: [Option<R>; D],
    ) -> Result<TensorRange<T, S, D>, InvalidShapeError<D>>
    where
        R: Into<IndexRange>,
    {
        TensorRange::clip_from(source, range.map(|option| option.map(|range| range.into())))
    }

    fn clip_from(
        source: S,
        range: [Option<IndexRange>; D],
    ) -> Result<TensorRange<T, S, D>, InvalidShapeError<D>> {
        let shape = source.view_shape();
        let mut range = {
            // TODO: A iterator enumerate call would be much cleaner here but everything
            // except array::map is not stable yet.
            let mut d = 0;
            range.map(|option| {
                // convert None to ranges that select the entire length of the tensor
                let range = option.unwrap_or_else(|| IndexRange::new(0, shape[d].1));
                d += 1;
                range
            })
        };
        let shape = InvalidShapeError {
            shape: clip_range_shape(&shape, &mut range),
        };
        if !shape.is_valid() {
            return Err(shape);
        }

        Ok(TensorRange {
            source,
            range,
            _type: PhantomData,
        })
    }

    /**
     * Constructs a TensorRange from a tensor and a range, ensuring the range is within the
     * lengths of the tensor.
     *
     * Returns the Err variant if any dimension would have a length of 0 or any range extends
     * beyond the length of that dimension in the tensor.
     *
     */
    pub fn from_all_strict<R>(
        source: S,
        range: [Option<R>; D],
    ) -> Result<TensorRange<T, S, D>, StrictIndexRangeValidationError<D, D>>
    where
        R: Into<IndexRange>,
    {
        let shape = source.view_shape();
        let range = range.map(|option| option.map(|range| range.into()));
        if range_exceeds_bounds(&shape, &range) {
            return Err(StrictIndexRangeValidationError::OutsideShape {
                shape,
                index_range: range,
            });
        }

        match TensorRange::clip_from(source, range) {
            Ok(tensor_range) => Ok(tensor_range),
            Err(invalid_shape) => Err(
                StrictIndexRangeValidationError::Error(
                    IndexRangeValidationError::InvalidShape(invalid_shape)
                )
            ),
        }
    }
}

fn range_exceeds_bounds<const D: usize>(
    source: &[(Dimension, usize); D],
    range: &[Option<IndexRange>; D],
) -> bool {
    for (d, (_, end)) in source.iter().enumerate() {
        let end = *end;
        match &range[d] {
            None => continue,
            Some(range) => {
                let range_end = range.start + range.length;
                match range_end > end {
                    true => return true,
                    false => (),
                };
            }
        }
    }
    false
}

// Returns the shape the tensor's shape will be left as with the range applied, clipping any
// ranges that exceed the bounds of the tensor's shape.
fn clip_range_shape<const D: usize>(
    source: &[(Dimension, usize); D],
    range: &mut [IndexRange; D],
) -> [(Dimension, usize); D] {
    let mut shape = *source;
    for (d, (_, length)) in shape.iter_mut().enumerate() {
        let _start = 0;
        let end = *length;
        let range = &range[d];
        let range_start = range.start;
        let range_end = range.start + range.length;
        let range_end = std::cmp::min(range_end, end);
        *length = range_end - range_start;
    }
    shape
}

impl<T, S, const D: usize> TensorMask<T, S, D>
where
    S: TensorRef<T, D>,
{
    // TODO: Alternate named constructors that take M dimension names and index ranges and call into
    // from/from_strict

    /**
     * Constructs a TensorMask from a tensor and a mask.
     *
     * Returns the Err variant if any masked dimension would have a length of 0.
     */
    pub fn from_all<R>(
        source: S,
        mask: [Option<R>; D]
    ) -> Result<TensorMask<T, S, D>, InvalidShapeError<D>>
    where
        R: Into<IndexRange>,
    {
        TensorMask::clip_from(source, mask.map(|option| option.map(|mask| mask.into())))
    }

    fn clip_from(
        source: S,
        mask: [Option<IndexRange>; D],
    ) -> Result<TensorMask<T, S, D>, InvalidShapeError<D>> {
        let shape = source.view_shape();
        let mut mask = mask.map(|option| option.unwrap_or_else(|| IndexRange::new(0, 0)));
        let shape = InvalidShapeError {
            shape: clip_masked_shape(&shape, &mut mask),
        };
        if !shape.is_valid() {
            return Err(shape);
        }

        Ok(TensorMask {
            source,
            mask,
            _type: PhantomData,
        })
    }

    /**
     * Constructs a TensorMask from a tensor and a mask, ensuring the mask is within range of
     * the tensor.
     *
     * Returns the Err variant if any masked dimension would have a length of 0 or any mask
     * extends beyond the length of that dimension in the tensor.
     *
     */
    pub fn from_strict<R>(
        source: S,
        mask: [Option<R>; D],
    ) -> Result<TensorMask<T, S, D>, StrictIndexRangeValidationError<D, D>>
    where
        R: Into<IndexRange>,
    {
        let shape = source.view_shape();
        let mask = mask.map(|option| option.map(|mask| mask.into()));
        if mask_exceeds_bounds(&shape, &mask) {
            return Err(StrictIndexRangeValidationError::OutsideShape {
                shape,
                index_range: mask,
            });
        }

        match TensorMask::clip_from(source, mask) {
            Ok(tensor_mask) => Ok(tensor_mask),
            Err(invalid_shape) => Err(
                StrictIndexRangeValidationError::Error(
                    IndexRangeValidationError::InvalidShape(invalid_shape)
                )
            ),
        }
    }
}

// Returns the shape the tensor's shape will be left as with the mask applied, clipping any
// masks that exceed the bounds of the tensor's shape.
fn clip_masked_shape<const D: usize>(
    source: &[(Dimension, usize); D],
    mask: &mut [IndexRange; D],
) -> [(Dimension, usize); D] {
    let mut shape = *source;
    for (d, (_, length)) in shape.iter_mut().enumerate() {
        let start = 0;
        let end = *length;
        let mask = &mask[d];
        let mask_start = mask.start;
        let mask_end = mask.start + mask.length;
        let mask_end = std::cmp::min(mask_end, end);
        let before_mask = mask_start - start;
        let after_mask = mask_end - end;
        *length = before_mask + after_mask;
    }
    shape
}

fn mask_exceeds_bounds<const D: usize>(
    source: &[(Dimension, usize); D],
    mask: &[Option<IndexRange>; D],
) -> bool {
    // same test for a mask extending past a shape as for a range
    range_exceeds_bounds(source, mask)
}
