# Migrating from version 1 to version 2

Version 2.0 of Easy ML contains several breaking changes, however if the code being updated does not add any Easy ML trait implementations it will likely be able to update with no code changes required:

## MatrixRef now has NoInteriorMutability as a supertrait.
This will ensure consumers migrating from 1.x versions see compile errors where
the new requirements will have been added. It is no longer possible to implement
a `MatrixRef` type that does not conform to the no interior mutability contract.

Migrating requires adding `NoInteriorMutability` implementations for any custom
matrix types that were previously only implementing `MatrixRef`. If you have
not written any custom matrix types, this change does not require any action.

## Version 2 adds blanket impls for & and &mut references to MatrixRef and MatrixMut.
Any manual implementations of reference types implementing these traits need to
be deleted as they are now longer required and conflict with the blanket impls.
If you have not written any custom matrix types, this change does not require
any action.

## Numeric now requires implementing types to implement Debug.
If you have implemented `Numeric` for any types, you will need to add a `Debug`
implementation if it was missing.

## Real and RealByValue now inherit from the corresponding Numeric and NumericByValue traits.
This means old code depending on a previous version of Easy ML that also
specified the `Numeric` traits such as:
```rust
fn function_name<T: Numeric + Real>()
where for<'a> &'a T: NumericRef<T> + RealRef<T> {}
```
can be updated when using Easy ML 2.0 or later to the following:
```rust
fn function_name<T: Real>()
where for<'a> &'a T: RealRef<T> {}
```
However, the old code will still compile correctly without any action.

## Private properties mean and covariance on the MultivariateGaussian struct
The public properties `mean` and `covariance` on the `MultivariateGaussian` struct were made private and methods with the same names were added to return
references to the vector and matrix. This allows the `draw` method to not have
to recheck invariants every time it is called, now matching the
`MultivariateGaussianTensor` version.
