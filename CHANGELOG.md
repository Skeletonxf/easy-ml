# Changelog

## Version 1.10

Still in development

## Versions 1.9.1 and 1.8.3

Both versions contain a fix for the `WithIndex` matrix row/column major
iterators not delegating to their base iterator exact size implementation.
Calling the `len()` methods on earlier versions will panic as the standard
library `len()` implementation for `ExactSizeIterator` checks the
invariant that the affected `WithIndex` iterators accidentally did not uphold.
The `len()` methods on the base iterators prior wrapping them in `WithIndex`
was correct and can be used to get the exact length if needed on earlier
versions of Easy ML.

Additionally version 1.9.1 includes `TensorView` implementing `Clone` where
applicable. On earlier versions `map` with a no-op closure can be used as a
partial workaround to return a `Tensor` with cloned data, which if needed could
be converted back to a `TensorView` with `from`.

Version 1.10 will also include both of these backported bugfixes.

## Version 1.9

Release of named Tensor APIs, and extended linear algebra support. Fixed serde
deserialisation issue with Matrices not validating their inputs. Fixed oversight
in Matrix Display impls where the default precision was truncating strings - now
precision is not defaulted to any value so no unexpected truncation will happen
automatically.

## Versions 1.8.2 and 1.7.2

Backported a series of bugfixes for the `MatrixRange` and `IndexRange` APIs.
Versions of Easy ML prior to 1.7 were not affected as the APIs did not exist
yet. `MatrixRange` did not properly clip the `IndexRanges` it was constructed
from to keep its `view_rows` and `view_columns` reporting the correct lengths.
`IndexRange` conversion methods from `[usize; 2]` and `(usize, usize)` have
been corrrected to match their documented behaviour. The `Range<usize>`
conversion method will now correctly saturate to 0 lengths if the `end` of the
range is equal to or less than the `start`.

Version 1.9 also includes all backported bugfixes since version 1.8.0

## Versions 1.8.1, 1.7.1 and 1.6.2

Backported a bugfix for `Matrix::transpose_mut` that affected all prior versions
of Easy ML. `Matrix::transpose_mut` did not do the correct thing if given non
square matrices and could have caused a panic or the matrix to be invalid. The
documented use on square matrices was correct, and works correctly on all
prior versions. If for some reason a user is stuck on a `1.5` or older version
of Easy ML, they can still check themselves that the matrix is square before
calling `Matrix::transpose_mut` and use `Matrix::transpose` instead if it is not.

Version 1.9 when eventually released will also include this bugfix.

## Version 1.8

Added mutable reference APIs for iterators, matrices and matrix views. Added
a matrix quadrants API which allows safely splitting a matrix into multiple
mutable parts. Added many numerical operations for MatrixViews to bring them to
feature parity with Matrices.

Named Tensors have started development but are not public API yet and not yet
available to use.

The project is now also formatted by `rustfmt`.

## Version 1.7

Added diagonal iterators. Added MatrixView, MatrixRef and MatrixMut APIs. Made
all matrix iterators generic over their source, allowing them to also be used
with matrix views. Added unsafe getters to elide bounds checks for Matrices, and
these are now used internally by matrix iterators.

## Version 1.6.1

Fixed README versions still referring to 1.5

## Version 1.6

Added QR decomposition function. Improved documentation examples using rand.
Added `#[track_caller]` to some functions that were missed in 1.5. Added
size_hint implementations to all the matrix iterators.

## Version 1.5

Added opt in serde support for `Matrix` and `Trace` behind the `serde` feature.
Improved documentation in various places, updated to use inter doc links,
updated the versions of dependencies to latest version, and added
`#[track_caller]` to many functions that could panic to provide better error
messages.

## Version 1.4

Flattened the internal storage of data in Matrices. This will make further
library changes easier in the future, and may provide a slight performance
improvement. The caveat to this change is that code which extensively adds
or removes columns from Matrices may run more slowly, however code which
reads or writes data in a row major format should be much more cache friendly.

Added explicit support for Web Assembly going forward, with a supporting
example on the MNIST dataset.

## Version 1.3.1

Fixed broken link to XOR example in documentation.

## Version 1.3

Added Forward and Reverse mode Automatic Differentiation wrapper structs.  
Added example for solving the XOR problem with a neural net using Automatic
Differentiation.  
Added RowMajor versions for matrix iterators  
Added matrix and scalar operations to Matrix  

## Version 1.2

Added a simpler Naïve Bayes example and supporting library code for
computing f1 scores.

## Version 1.1

Deprecated `Matrix::unit` and renamed to `Matrix::from_scalar`
Deprecated `Gaussian::map` and renamed to `Gaussian::probability`

Added fully worked Naïve Bayes example and supporting library code for
Gaussians and linear algebra.

Improved the explanation in some of the runtime `panic!` errors

## Version 1

Released with examples and library code for:

- Linear Regression
- k-means Clustering
- Logistic Regression
- using a custom numeric type such as `num_bigint::BigInt`
