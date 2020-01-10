/*!
Using custom numeric types examples.

# Using a custom numeric type

The following example shows how to make a type defined outside this crate implement Numeric
so it can be used by this library for matrix operations.

In this case the type is `BigInt` from `num_bigint` but as this type is also
outside this crate we need to wrap it in another struct first so we can
implement traits on it due to Rust's orphan rules.

```
extern crate easy_ml;
extern crate num_bigint;
extern crate num_traits;

use std::ops::{Add, Sub, Mul, Div, Neg};
use std::iter::Sum;

use easy_ml::numeric::{ZeroOne, FromUsize};
use easy_ml::matrices::Matrix;
use num_bigint::{BigInt, ToBigInt, Sign};
use num_traits::{Zero, One};

#[derive(Clone, Eq, PartialEq, PartialOrd, Debug)]
struct BigIntWrapper(BigInt);

/*
 * First we define a utility function to obtain the &BigInt
 * from a &BigIntWrapper as we don't want to have to clone
 * the BigInt when we can avoid it.
 */

impl BigIntWrapper {
    /**
     * A utility function to get the &BigInt from a &BigIntWrapper
     */
    fn unwrap(&self) -> &BigInt {
        let BigIntWrapper(ref x) = *self;
        x
    }
}

/*
 * Then implement the two traits defined in easy_ml onto the wrapper.
 */

impl ZeroOne for BigIntWrapper {
    #[inline]
    fn zero() -> BigIntWrapper {
        BigIntWrapper(Zero::zero())
    }
    #[inline]
    fn one() -> BigIntWrapper {
        BigIntWrapper(One::one())
    }
}

impl FromUsize for BigIntWrapper {
    #[inline]
    fn from_usize(n: usize) -> Option<BigIntWrapper> {
        let bigint = ToBigInt::to_bigint(&n)?;
        Some(BigIntWrapper(bigint))
    }
}

/*
 * If we weren't wrapping BigInt and could have defined `ZeroOne`
 * and `FromUsize` directly on it we would stop here. As we have
 * to wrap it, all that's left is a lot of boilerplate to
 * redefine most of the traits BigInt already implements on our
 * wrapper.
 */

// Define 4 macros to implement the operations Add, Sub, Mul and Div for
// all 4 combinations of by value and by reference BigIntWrappers

// BigIntWrapper op BigIntWrapper
macro_rules! operation_impl_value_value {
    (impl $Trait:ident for BigIntWrapper { fn $method:ident }) => {
        impl $Trait<BigIntWrapper> for BigIntWrapper {
            type Output = BigIntWrapper;

            #[inline]
            fn $method(self, rhs: BigIntWrapper) -> Self::Output {
                // unwrap, do operation unwrapped, then wrap output
                BigIntWrapper((self.0).$method(rhs.0))
            }
        }
    }
}

// BigIntWrapper op &BigIntWrapper
macro_rules! operation_impl_value_reference {
    (impl $Trait:ident for BigIntWrapper { fn $method:ident }) => {
        impl <'a> $Trait<&'a BigIntWrapper> for BigIntWrapper {
            type Output = BigIntWrapper;

            #[inline]
            fn $method(self, rhs: &BigIntWrapper) -> Self::Output {
                // unwrap, do operation unwrapped, then wrap output
                BigIntWrapper((self.0).$method(rhs.unwrap()))
            }
        }
    }
}

// &BigIntWrapper op BigIntWrapper
macro_rules! operation_impl_reference_value {
    (impl $Trait:ident for BigIntWrapper { fn $method:ident }) => {
        impl <'a> $Trait<BigIntWrapper> for &'a BigIntWrapper {
            type Output = BigIntWrapper;

            #[inline]
            fn $method(self, rhs: BigIntWrapper) -> Self::Output {
                // unwrap, do operation unwrapped, then wrap output
                BigIntWrapper(self.unwrap().$method(rhs.0))
            }
        }
    }
}

// &BigIntWrapper op &BigIntWrapper
macro_rules! operation_impl_reference_reference {
    (impl $Trait:ident for BigIntWrapper { fn $method:ident }) => {
        impl <'a, 'b> $Trait<&'a BigIntWrapper> for &'b BigIntWrapper {
            type Output = BigIntWrapper;

            #[inline]
            fn $method(self, rhs: &BigIntWrapper) -> Self::Output {
                // unwrap, do operation unwrapped, then wrap output
                BigIntWrapper(self.unwrap().$method(rhs.unwrap()))
            }
        }
    }
}

// Now we can implement these operations for Add, Sub, Mul and Div in one go
// instead of writing them out 4 times each.
operation_impl_value_value! { impl Add for BigIntWrapper { fn add } }
operation_impl_value_value! { impl Sub for BigIntWrapper { fn sub } }
operation_impl_value_value! { impl Mul for BigIntWrapper { fn mul } }
operation_impl_value_value! { impl Div for BigIntWrapper { fn div } }
operation_impl_value_reference! { impl Add for BigIntWrapper { fn add } }
operation_impl_value_reference! { impl Sub for BigIntWrapper { fn sub } }
operation_impl_value_reference! { impl Mul for BigIntWrapper { fn mul } }
operation_impl_value_reference! { impl Div for BigIntWrapper { fn div } }
operation_impl_reference_value! { impl Add for BigIntWrapper { fn add } }
operation_impl_reference_value! { impl Sub for BigIntWrapper { fn sub } }
operation_impl_reference_value! { impl Mul for BigIntWrapper { fn mul } }
operation_impl_reference_value! { impl Div for BigIntWrapper { fn div } }
operation_impl_reference_reference! { impl Add for BigIntWrapper { fn add } }
operation_impl_reference_reference! { impl Sub for BigIntWrapper { fn sub } }
operation_impl_reference_reference! { impl Mul for BigIntWrapper { fn mul } }
operation_impl_reference_reference! { impl Div for BigIntWrapper { fn div } }

/*
 * Because Neg is a unary operation there are only two combinations
 * so the implementations are written out longhand for ease of understanding.
 */

// - BigIntWrapper
impl Neg for BigIntWrapper {
    type Output = BigIntWrapper;

    #[inline]
    fn neg(self) -> Self::Output {
        // unwrap, do operation unwrapped, then wrap output
        BigIntWrapper(-(self.0))
    }
}

// - &BigIntWrapper
// Like with the macro'd versions for Add, Sub, Mul and Div, we need to define
// the Neg trait on a reference to a BigIntWrapper, which is itself a type.
impl <'a> Neg for &'a BigIntWrapper {
    type Output = BigIntWrapper;

    #[inline]
    fn neg(self) -> Self::Output {
        // unwrap, do operation unwrapped, then wrap output
        BigIntWrapper(-(self.unwrap()))
    }
}

// Finally we need to implement Sum on just types of BigIntWrapper
// to complete the steps to getting Numeric implemented on BigIntWrapper

impl Sum<BigIntWrapper> for BigIntWrapper {
    fn sum<I>(iter: I) -> BigIntWrapper
    where I: Iterator<Item = BigIntWrapper> {
        // unwrap every item in the iterator, do sum unwrapped, then wrap output
        BigIntWrapper(iter.map(|wrapped| wrapped.0).sum())
    }
}

// For convenience and demonstrations below ToString is also implemented on BigIntWrapper
impl ToString for BigIntWrapper {
    #[inline]
    fn to_string(&self) -> String {
        self.unwrap().to_string()
    }
}

let one_million = ToBigInt::to_bigint(&1000000).unwrap();
let wrapped = BigIntWrapper(one_million);

let matrix = Matrix::unit(wrapped);
println!("1000000 x 1000000 = {:?}", (&matrix * &matrix).get_reference(0, 0).to_string());

// Wrapping and unwrapping transformations can be done with map
let unwrapped: Matrix<BigInt> = matrix.map(|wrapped| wrapped.0);
println!("Unwrapped:\n{:?}", unwrapped);
let matrix: Matrix<BigInt> = Matrix::unit(ToBigInt::to_bigint(&-3).unwrap());
let wrapped: Matrix<BigIntWrapper> = matrix.map(|unwrapped| BigIntWrapper(unwrapped));
println!("Wrapped:\n{:?}", wrapped);
```
*/
