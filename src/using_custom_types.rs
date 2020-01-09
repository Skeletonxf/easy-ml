/*!
Using custom numeric types examples.

# Using a custom numeric type

The following example shows how to make a type defined outside this crate implement Numeric
so it can be used by this library for matrix operations.

In this case the type is `BigInt` from `num_bigint` but as this type is also
outside this crate we need to wrap it in another struct first so we can
implement traits on it.

```
extern crate easy_ml;
extern crate num_bigint;
extern crate num_traits;

//use std::ops::Deref;

use easy_ml::numeric::{ZeroOne, FromUsize};
use easy_ml::matrices::Matrix;
use num_bigint::{BigInt, ToBigInt, Sign};
use num_traits::{Zero, One};

struct BigIntWrapper(BigInt);

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
// Check if this helps at all
// impl Deref for BigIntWrapper {
//     type Target = BigInt;
//
//     fn deref(&self) -> &BigInt {
//         &self.0
//     }
// }

let five = ToBigInt::to_bigint(&5).unwrap();
let wrapped_five = BigIntWrapper(five);
// let five_times_five = &wrapped_five * &wrapped_five;
// println!("5 x 5: {}", five_times_five);

// let matrix = Matrix::unit(wrapped_five);
// println!("5 x 5: {:?}", &matrix * &matrix);
```
*/
