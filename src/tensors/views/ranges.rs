use crate::tensors::views::{TensorRef, TensorMut};
use crate::tensors::{Dimension, InvalidDimensionsError, InvalidShapeError};
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use crate::matrices::views::IndexRange;

// TODO: Test all the TensorRef/TensorMut implemenations and constructor validation

/**
 * A range over a tensor in D dimensions, hiding the values **outside** the range from view.
 *
 * The entire source is still owned by the TensorRange however, so this does not permit
 * creating multiple mutable ranges into a single tensor even if they wouldn't overlap.
 *
 * See also: [TensorMask](TensorMask)
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
 *
 * See also: [TensorRange](TensorRange)
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

impl<const D: usize, const P: usize> fmt::Display for IndexRangeValidationError<D, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexRangeValidationError::InvalidShape(error) => write!(f, "{:?}", error),
            IndexRangeValidationError::InvalidDimensions(error) => write!(f, "{:?}", error),
        }
    }
}

impl<const D: usize, const P: usize> Error for IndexRangeValidationError<D, P> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            IndexRangeValidationError::InvalidShape(error) => Some(error),
            IndexRangeValidationError::InvalidDimensions(error) => Some(error),
        }
    }
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

impl<const D: usize, const P: usize> fmt::Display for StrictIndexRangeValidationError<D, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use StrictIndexRangeValidationError as S;
        match self {
            S::OutsideShape { shape, index_range } => write!(
                f,
                "IndexRange array {:?} is out of bounds of shape {:?}",
                index_range, shape
            ),
            S::Error(error) => write!(f, "{:?}", error),
        }
    }
}

impl<const D: usize, const P: usize> Error for StrictIndexRangeValidationError<D, P> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        use StrictIndexRangeValidationError as S;
        match self {
            S::OutsideShape {
                shape: _,
                index_range: _,
            } => None,
            S::Error(error) => Some(error),
        }
    }
}

fn from_named_to_all<T, S, R, const D: usize, const P: usize>(
    source: &S,
    ranges: [(Dimension, R); P],
) -> Result<[Option<IndexRange>; D], IndexRangeValidationError<D, P>>
where
    S: TensorRef<T, D>,
    R: Into<IndexRange>,
{
    let shape = source.view_shape();
    let ranges = ranges.map(|(d, r)| (d, r.into()));
    let dimensions = InvalidDimensionsError {
        provided: ranges.clone().map(|(d, _)| d),
        valid: shape.map(|(d, _)| d),
    };
    if dimensions.has_duplicates() {
        return Err(IndexRangeValidationError::InvalidDimensions(dimensions));
    }
    // Since we now know there's no duplicates, we can lookup the dimension index for each name
    // in the shape and we know we'll get different indexes on each lookup.
    let mut all_ranges: [Option<IndexRange>; D] = [(); D].map(|_| None);
    for (name, range) in ranges.into_iter() {
        match crate::tensors::dimensions::position_of(&shape, name) {
            Some(d) => all_ranges[d] = Some(range),
            None => return Err(IndexRangeValidationError::InvalidDimensions(dimensions)),
        };
    }
    Ok(all_ranges)
}

impl<T, S, const D: usize> TensorRange<T, S, D>
where
    S: TensorRef<T, D>,
{
    /**
     * Constructs a TensorRange from a tensor and set of dimension name/range pairs.
     *
     * Returns the Err variant if any dimension would have a length of 0 after applying the
     * ranges, if multiple pairs with the same name are provided, or if any dimension names aren't
     * in the source.
     */
    pub fn from<R, const P: usize>(
        source: S,
        ranges: [(Dimension, R); P],
    ) -> Result<TensorRange<T, S, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        let all_ranges = from_named_to_all(&source, ranges)?;
        match TensorRange::from_all(source, all_ranges) {
            Ok(tensor_range) => Ok(tensor_range),
            Err(invalid_shape) => Err(IndexRangeValidationError::InvalidShape(invalid_shape)),
        }
    }

    /**
     * Constructs a TensorRange from a tensor and set of dimension name/range pairs.
     *
     * Returns the Err variant if any dimension would have a length of 0 after applying the
     * ranges, if multiple pairs with the same name are provided, or if any dimension names aren't
     * in the source, or any range extends beyond the length of that dimension in the tensor.
     */
    pub fn from_strict<R, const P: usize>(
        source: S,
        ranges: [(Dimension, R); P],
    ) -> Result<TensorRange<T, S, D>, StrictIndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        use StrictIndexRangeValidationError as S;
        let all_ranges = match from_named_to_all(&source, ranges) {
            Ok(all_ranges) => all_ranges,
            Err(error) => return Err(S::Error(error)),
        };
        match TensorRange::from_all_strict(source, all_ranges) {
            Ok(tensor_range) => Ok(tensor_range),
            Err(S::OutsideShape { shape, index_range }) => Err(
                S::OutsideShape { shape, index_range }
            ),
            Err(S::Error(IndexRangeValidationError::InvalidShape(error))) => Err(
                S::Error(IndexRangeValidationError::InvalidShape(error))
            ),
            Err(S::Error(IndexRangeValidationError::InvalidDimensions(_))) => panic!(
                "Unexpected InvalidDimensions error case after validating for InvalidDimensions already"
            ),
        }
    }

    /**
     * Constructs a TensorRange from a tensor and a range for each dimension in the tensor
     * (provided in the same order as the tensor's shape).
     *
     * Returns the Err variant if any dimension would have a length of 0 after applying the ranges.
     */
    pub fn from_all<R>(
        source: S,
        ranges: [Option<R>; D],
    ) -> Result<TensorRange<T, S, D>, InvalidShapeError<D>>
    where
        R: Into<IndexRange>,
    {
        TensorRange::clip_from(
            source,
            ranges.map(|option| option.map(|range| range.into())),
        )
    }

    fn clip_from(
        source: S,
        ranges: [Option<IndexRange>; D],
    ) -> Result<TensorRange<T, S, D>, InvalidShapeError<D>> {
        let shape = source.view_shape();
        let mut ranges = {
            // TODO: A iterator enumerate call would be much cleaner here but everything
            // except array::map is not stable yet.
            let mut d = 0;
            ranges.map(|option| {
                // convert None to ranges that select the entire length of the tensor
                let range = option.unwrap_or_else(|| IndexRange::new(0, shape[d].1));
                d += 1;
                range
            })
        };
        let shape = InvalidShapeError {
            shape: clip_range_shape(&shape, &mut ranges),
        };
        if !shape.is_valid() {
            return Err(shape);
        }

        Ok(TensorRange {
            source,
            range: ranges,
            _type: PhantomData,
        })
    }

    /**
     * Constructs a TensorRange from a tensor and a range for each dimension in the tensor
     * (provided in the same order as the tensor's shape), ensuring the range is within the
     * lengths of the tensor.
     *
     * Returns the Err variant if any dimension would have a length of 0 after applying the
     * ranges or any range extends beyond the length of that dimension in the tensor.
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
            Err(invalid_shape) => Err(StrictIndexRangeValidationError::Error(
                IndexRangeValidationError::InvalidShape(invalid_shape),
            )),
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
    /**
     * Constructs a TensorMask from a tensor and set of dimension name/mask pairs.
     *
     * Returns the Err variant if any masked dimension would have a length of 0, if multiple
     * pairs with the same name are provided, or if any dimension names aren't in the source.
     */
    pub fn from<R, const P: usize>(
        source: S,
        masks: [(Dimension, R); P],
    ) -> Result<TensorMask<T, S, D>, IndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        let all_masks = from_named_to_all(&source, masks)?;
        match TensorMask::from_all(source, all_masks) {
            Ok(tensor_mask) => Ok(tensor_mask),
            Err(invalid_shape) => Err(IndexRangeValidationError::InvalidShape(invalid_shape)),
        }
    }

    /**
     * Constructs a TensorMask from a tensor and set of dimension name/range pairs.
     *
     * Returns the Err variant if any masked dimension would have a length of 0, if multiple
     * pairs with the same name are provided, or if any dimension names aren't in the source,
     * or any mask extends beyond the length of that dimension in the tensor.
     */
    pub fn from_strict<R, const P: usize>(
        source: S,
        masks: [(Dimension, R); P],
    ) -> Result<TensorMask<T, S, D>, StrictIndexRangeValidationError<D, P>>
    where
        R: Into<IndexRange>,
    {
        use StrictIndexRangeValidationError as S;
        let all_masks = match from_named_to_all(&source, masks) {
            Ok(all_masks) => all_masks,
            Err(error) => return Err(S::Error(error)),
        };
        match TensorMask::from_all_strict(source, all_masks) {
            Ok(tensor_mask) => Ok(tensor_mask),
            Err(S::OutsideShape { shape, index_range }) => Err(
                S::OutsideShape { shape, index_range }
            ),
            Err(S::Error(IndexRangeValidationError::InvalidShape(error))) => Err(
                S::Error(IndexRangeValidationError::InvalidShape(error))
            ),
            Err(S::Error(IndexRangeValidationError::InvalidDimensions(_))) => panic!(
                "Unexpected InvalidDimensions error case after validating for InvalidDimensions already"
            ),
        }
    }

    /**
     * Constructs a TensorMask from a tensor and a mask for each dimension in the tensor
     * (provided in the same order as the tensor's shape).
     *
     * Returns the Err variant if any masked dimension would have a length of 0.
     */
    pub fn from_all<R>(
        source: S,
        mask: [Option<R>; D],
    ) -> Result<TensorMask<T, S, D>, InvalidShapeError<D>>
    where
        R: Into<IndexRange>,
    {
        TensorMask::clip_from(source, mask.map(|option| option.map(|mask| mask.into())))
    }

    fn clip_from(
        source: S,
        masks: [Option<IndexRange>; D],
    ) -> Result<TensorMask<T, S, D>, InvalidShapeError<D>> {
        let shape = source.view_shape();
        let mut masks = masks.map(|option| option.unwrap_or_else(|| IndexRange::new(0, 0)));
        let shape = InvalidShapeError {
            shape: clip_masked_shape(&shape, &mut masks),
        };
        if !shape.is_valid() {
            return Err(shape);
        }

        Ok(TensorMask {
            source,
            mask: masks,
            _type: PhantomData,
        })
    }

    /**
     * Constructs a TensorMask from a tensor and a mask for each dimension in the tensor
     * (provided in the same order as the tensor's shape), ensuring the mask is within the
     * lengths of the tensor.
     *
     * Returns the Err variant if any masked dimension would have a length of 0 or any mask
     * extends beyond the length of that dimension in the tensor.
     */
    pub fn from_all_strict<R>(
        source: S,
        masks: [Option<R>; D],
    ) -> Result<TensorMask<T, S, D>, StrictIndexRangeValidationError<D, D>>
    where
        R: Into<IndexRange>,
    {
        let shape = source.view_shape();
        let masks = masks.map(|option| option.map(|mask| mask.into()));
        if mask_exceeds_bounds(&shape, &masks) {
            return Err(StrictIndexRangeValidationError::OutsideShape {
                shape,
                index_range: masks,
            });
        }

        match TensorMask::clip_from(source, masks) {
            Ok(tensor_mask) => Ok(tensor_mask),
            Err(invalid_shape) => Err(StrictIndexRangeValidationError::Error(
                IndexRangeValidationError::InvalidShape(invalid_shape),
            )),
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

fn map_indexes_by_range<const D: usize>(
    indexes: [usize; D],
    ranges: &[IndexRange; D]
) -> Option<[usize; D]> {
    let mut mapped = [0; D];
    for (d, (r, i)) in ranges.iter().zip(indexes.into_iter()).enumerate() {
        mapped[d] = r.map(i)?;
    }
    Some(mapped)
}

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// and just hiding some of the valid indexes from view, we implement TensorRef correctly as well.
/**
 * A TensorRange implements TensorRef, with the dimension lengths reduced to the range the
 * the TensorRange was created with.
 */
unsafe impl<T, S, const D: usize> TensorRef<T, D> for TensorRange<T, S, D>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.source.get_reference(map_indexes_by_range(indexes, &self.range)?)
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        // Since when we were constructed we clipped the length of each range to no more than
        // our source, we can just return the length of each range now
        let mut shape = self.source.view_shape();
        // TODO: zip would work really nicely here but it's not stable yet
        for (mut pair, range) in shape.iter_mut().zip(self.range.iter()) {
            pair.1 = range.length;
        }
        shape
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        // It is the caller's responsibiltiy to always call with indexes in range,
        // therefore the unwrap() case should never happen because on an arbitary TensorRef
        // it would be undefined behavior.
        // TODO: Can we use unwrap_unchecked here?
        self.source.get_reference_unchecked(map_indexes_by_range(indexes, &self.range).unwrap())
    }
}

// # Safety
//
// The type implementing TensorMut must implement it correctly, so by delegating to it
// and just hiding some of the valid indexes from view, we implement TensorMut correctly as well.
/**
 * A TensorRange implements TensorMut, with the dimension lengths reduced to the range the
 * the TensorRange was created with.
 */
unsafe impl<T, S, const D: usize> TensorMut<T, D> for TensorRange<T, S, D>
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.source.get_reference_mut(map_indexes_by_range(indexes, &self.range)?)
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        // It is the caller's responsibiltiy to always call with indexes in range,
        // therefore the unwrap() case should never happen because on an arbitary TensorMut
        // it would be undefined behavior.
        // TODO: Can we use unwrap_unchecked here?
        self.source.get_reference_unchecked_mut(
            map_indexes_by_range(indexes, &self.range).unwrap()
        )
    }
}

fn map_indexes_by_mask<const D: usize>(
    indexes: [usize; D],
    masks: &[IndexRange; D]
) -> [usize; D] {
    let mut mapped = [0; D];
    for (d, (r, i)) in masks.iter().zip(indexes.into_iter()).enumerate() {
        mapped[d] = r.mask(i);
    }
    mapped
}

// # Safety
//
// The type implementing TensorRef must implement it correctly, so by delegating to it
// and just hiding some of the valid indexes from view, we implement TensorRef correctly as well.
/**
 * A TensorMask implements TensorRef, with the dimension lengths reduced by the mask the
 * the TensorMask was created with.
 */
unsafe impl<T, S, const D: usize> TensorRef<T, D> for TensorMask<T, S, D>
where
    S: TensorRef<T, D>,
{
    fn get_reference(&self, indexes: [usize; D]) -> Option<&T> {
        self.source.get_reference(map_indexes_by_mask(indexes, &self.mask))
    }

    fn view_shape(&self) -> [(Dimension, usize); D] {
        // Since when we were constructed we clipped the length of each mask to no more than
        // our source, we can just return subtract length of each mask now
        let mut shape = self.source.view_shape();
        // TODO: zip would work really nicely here but it's not stable yet
        for (mut pair, mask) in shape.iter_mut().zip(self.mask.iter()) {
            pair.1 -= mask.length;
        }
        shape
    }

    unsafe fn get_reference_unchecked(&self, indexes: [usize; D]) -> &T {
        // It is the caller's responsibiltiy to always call with indexes in range,
        // therefore out of bounds lookups created by map_indexes_by_mask should never happen.
        self.source.get_reference_unchecked(map_indexes_by_mask(indexes, &self.mask))
    }
}

// # Safety
//
// The type implementing TensorMut must implement it correctly, so by delegating to it
// and just hiding some of the valid indexes from view, we implement TensorMut correctly as well.
/**
 * A TensorMask implements TensorMut, with the dimension lengths reduced by the mask the
 * the TensorMask was created with.
 */
unsafe impl<T, S, const D: usize> TensorMut<T, D> for TensorMask<T, S, D>
where
    S: TensorMut<T, D>,
{
    fn get_reference_mut(&mut self, indexes: [usize; D]) -> Option<&mut T> {
        self.source.get_reference_mut(map_indexes_by_mask(indexes, &self.mask))
    }

    unsafe fn get_reference_unchecked_mut(&mut self, indexes: [usize; D]) -> &mut T {
        // It is the caller's responsibiltiy to always call with indexes in range,
        // therefore out of bounds lookups created by map_indexes_by_mask should never happen.
        self.source.get_reference_unchecked_mut(map_indexes_by_mask(indexes, &self.mask))
    }
}
