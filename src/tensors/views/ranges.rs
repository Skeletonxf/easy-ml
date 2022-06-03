use crate::tensors::views::TensorRef;
use crate::tensors::{Dimension, InvalidShapeError};
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
    range: [Option<IndexRange>; D], // TODO: We can probably coerce None to IndexRanges that cover start to end to save a condition when indexing
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
    mask: [Option<IndexRange>; D], // TODO: We can probably coerce None to IndexRanges that have a length of 0 to save a condition when indexing
    _type: PhantomData<T>,
}

// TODO: Document this
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IndexRangeValidationError<const D: usize> {
    InvalidMaskedShape(InvalidShapeError<D>),
    RangeOutsideShape {
        shape: [(Dimension, usize); D],
        index_range: [Option<IndexRange>; D],
    }
}

impl<T, S, const D: usize> TensorRange<T, S, D>
where
    S: TensorRef<T, D>,
{
    // TODO: Alternate named constructors that take M dimension names and index ranges and call into
    // from/from_strict

    /**
     * Constructs a TensorRange from a tensor and a range.
     *
     * Returns the Err variant if any dimension would have a length of 0.
     */
    fn from<R>(
        source: S,
        range: [Option<R>; D],
    ) -> Result<TensorRange<T, S, D>, InvalidShapeError<D>>
    where
        R: Into<IndexRange>,
    {
        let range = range.map(|option| option.map(|range| range.into()));
        let shape = InvalidShapeError {
            shape: range_shape(&source.view_shape(), &range),
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
    fn from_strict<R>(
        source: S,
        range: [Option<R>; D],
    ) -> Result<TensorRange<T, S, D>, IndexRangeValidationError<D>>
    where
        R: Into<IndexRange>,
    {
        unimplemented!()
    }
}

// TODO: Maybe we should convert Option<IndexRange> to IndexRange and clamp the range to match our lengths so the range is our shape?
fn range_shape<const D: usize>(source: &[(Dimension, usize); D], range: &[Option<IndexRange>; D]) -> [(Dimension, usize); D] {
    let mut shape = *source;
    for (d, (_, length)) in shape.iter_mut().enumerate() {
        let start = 0;
        let end = source[d].1;
        match &range[d] {
            None => continue,
            Some(range) => {
                let range_start = range.start;
                let range_end = range.start + range.length;
                let range_end = std::cmp::min(range_end, end);
                *length = range_end - range_start;
            },
        };
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
    fn from<R>(
        source: S,
        mask: [Option<R>; D],
    ) -> Result<TensorMask<T, S, D>, InvalidShapeError<D>>
    where
        R: Into<IndexRange>,
    {
        let mask = mask.map(|option| option.map(|range| range.into()));
        let shape = InvalidShapeError {
            shape: masked_shape(&source.view_shape(), &mask),
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
    fn from_strict<R>(
        source: S,
        mask: [Option<R>; D],
    ) -> Result<TensorMask<T, S, D>, IndexRangeValidationError<D>>
    where
        R: Into<IndexRange>,
    {
        match TensorMask::from(source, mask) {
            Err(invalid_shape) => Err(IndexRangeValidationError::InvalidMaskedShape(invalid_shape)),
            Ok(tensor_mask) => {
                let shape = tensor_mask.source.view_shape();
                let mask = tensor_mask.mask.clone();
                for d in 0..D {
                    let end = shape[d].1;
                    match &mask[d] {
                        None => continue,
                        Some(m) => {
                            let mask_start = m.start;
                            let mask_end = mask_start + m.length;
                            if mask_end > end {
                                return Err(IndexRangeValidationError::RangeOutsideShape {
                                    shape,
                                    index_range: mask,
                                })
                            }
                        },
                    };
                }
                Ok(tensor_mask)
            },
        }
    }
}

fn masked_shape<const D: usize>(source: &[(Dimension, usize); D], mask: &[Option<IndexRange>; D]) -> [(Dimension, usize); D] {
    let mut shape = *source;
    for (d, (_, length)) in shape.iter_mut().enumerate() {
        let start = 0;
        let end = source[d].1;
        match &mask[d] {
            None => continue,
            Some(mask) => {
                let mask_start = mask.start;
                let mask_end = mask.start + mask.length;
                let before_mask = mask_start - start;
                let after_mask = mask_end - end;
                *length = before_mask + after_mask;
            },
        };
    }
    shape
}
