/*!
 *
 * TODO
 */

use crate::numeric::{Numeric, NumericRef, ZeroOne, FromUsize};
use crate::numeric::extra::{Real, RealRef};
use std::ops::{Add, Sub, Mul, Neg, Div};
use std::cmp::Ordering;
use std::num::Wrapping;
use std::iter::Sum;


/**
 * A trait with no methods which is implemented for all primitive types.
 *
 * Importantly this trait is not implemented for Traces, to stop the compiler
 * from trying to evaluate nested Traces of Traces as Numeric types. There is no
 * reason to create a Trace of a Trace, it won't do anything a Trace can't except
 * use more memory.
 *
 * The boilerplate implementations for primitives is performed with a macro.
 * If a primitive type is missing from this list, please open an issue to add it in.
 */
pub trait Primitive {}

macro_rules! impl_primitive {
    ($T:tt) => {
        impl Primitive for $T {}
    }
}

impl_primitive!(u8);
impl_primitive!(i8);
impl_primitive!(u16);
impl_primitive!(i16);
impl_primitive!(u32);
impl_primitive!(i32);
impl_primitive!(u64);
impl_primitive!(i64);
impl_primitive!(u128);
impl_primitive!(i128);
impl_primitive!(f32);
impl_primitive!(f64);
impl_primitive!(usize);
impl_primitive!(isize);

impl <T: Primitive> Primitive for Wrapping<T> {}

/**
 * A dual number which traces a real number and keeps track of its derivative.
 * This is used to perform Automatic Differentiation, as in no getting out the paper
 * to do calculus with algebra!
 *
 * Trace implements only first order differentiation. For example, given a function
 * 3x<sup>2</sup>, you can use calculus to work out that its derivative with respect
 * to x is 6x. You can also take the derivative of 6x with respect to x and work out
 * that the second derivative is 6. By instead writing the function 3x<sup>2</sup> in
 * code using Trace types as your numbers you can compute the first order derivative
 * for a given value of x by passing your function `Trace { number: x, derivative: 1.0 }`.
 *
 * ```
 * use easy_ml::differentiation::Trace;
 * let x = Trace { number: 3.2, derivative: 1.0 };
 * let dx = Trace::constant(3.0) * x * x;
 * assert_eq!(dx.derivative, 3.2 * 6.0);
 * ```
 */
#[derive(Debug)]
pub struct Trace<T: Primitive> {
    /**
     * The real number
     */
    pub number: T,
    /**
     * The first order derivative of this number.
     */
    pub derivative: T
}

impl <T: Numeric + Primitive> Trace<T> {
    /**
     * Constants are lifted to Traces with a derivative of 0
     */
    pub fn constant(c: T) -> Trace<T> {
        Trace {
            number: c,
            derivative: T::zero(),
        }
    }

    /**
     * To lift a variable that you want to find the derivative of
     * a function to, the Trace starts with a derivative of 1
     */
    pub fn variable(x: T) -> Trace<T> {
        Trace {
            number: x,
            derivative: T::one(),
        }
    }

    /**
     * Computes the derivative of a function with respect to its input x.
     *
     * This is a shorthand for `(function(Trace::variable(x))).derivative`
     */
    pub fn derivative(function: impl Fn(Trace<T>) -> Trace<T>, x: T) -> T {
        (function(Trace::variable(x))).derivative
    }
}

impl <T: Numeric + Primitive> ZeroOne for Trace<T> {
    #[inline]
    fn zero() -> Trace<T> {
        Trace::constant(T::zero())
    }
    #[inline]
    fn one() -> Trace<T> {
        Trace::constant(T::one())
    }
}

impl <T: Numeric + Primitive> FromUsize for Trace<T> {
    #[inline]
    fn from_usize(n: usize) -> Option<Trace<T>> {
        Some(Trace::constant(T::from_usize(n)?))
    }
}

/**
 * Any trace of a Cloneable type implements clone
 */
impl <T: Clone + Primitive> Clone for Trace<T> {
    fn clone(&self) -> Self {
        Trace {
            number: self.number.clone(),
            derivative: self.derivative.clone(),
        }
    }
}

/**
 * Any trace of a Copy type implements Copy
 */
impl <T: Copy + Primitive> Copy for Trace<T> { }

/**
 * Any trace of a PartialEq type implements PartialEq
 *
 * Note that as a Trace is intended to be substitutable with its
 * type T only the number parts of the trace are compared.
 * Hence the following is true
 * ```
 * use easy_ml::differentiation::Trace;
 * assert_eq!(Trace { number: 0, derivative: 1 }, Trace { number: 0, derivative: 2 })
 * ```
 */
impl <T: PartialEq + Primitive> PartialEq for Trace<T> {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

/**
 * Any trace of a PartialOrd type implements PartialOrd
 *
 * Note that as a Trace is intended to be substitutable with its
 * type T only the number parts of the trace are compared.
 * Hence the following is true
 * ```
 * use easy_ml::differentiation::Trace;
 * assert!(Trace { number: 1, derivative: 1 } > Trace { number: 0, derivative: 2 })
 * ```
 */
impl <T: PartialOrd + Primitive> PartialOrd for Trace<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

/**
 * Any trace of a Numeric type implements Sum, which is
 * the same as adding a bunch of Trace types together.
 */
impl <T: Numeric + Primitive> Sum for Trace<T> {
    fn sum<I>(mut iter: I) -> Trace<T> where I: Iterator<Item = Trace<T>> {
        let mut total = Trace::<T>::zero();
        loop {
            match iter.next() {
                None => return total,
                Some(next) => total = Trace {
                    number: total.number + next.number,
                    derivative: total.derivative + next.derivative
                },
            }
        }
    }
}

/**
 * Addition for two traces of the same type with both referenced.
 */
impl <T: Numeric + Primitive> Add for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    #[inline]
    fn add(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone() + rhs.number.clone(),
            derivative: self.derivative.clone() + rhs.derivative.clone(),
        }
    }
}

macro_rules! operator_impl_value_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
        * Operation for two traces of the same type.
        */
        impl <T: Numeric + Primitive> $op for Trace<T>
        where for<'a> &'a T: NumericRef<T> {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: Trace<T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    }
}

macro_rules! operator_impl_value_reference {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
        * Operation for two traces of the same type with the right referenced.
        */
        impl <T: Numeric + Primitive> $op<&Trace<T>> for Trace<T>
        where for<'a> &'a T: NumericRef<T> {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: &Trace<T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    }
}

macro_rules! operator_impl_reference_value {
    (impl $op:tt for Trace { fn $method:ident }) => {
        /**
        * Operation for two traces of the same type with the left referenced.
        */
        impl <T: Numeric + Primitive> $op<Trace<T>> for &Trace<T>
        where for<'a> &'a T: NumericRef<T> {
            type Output = Trace<T>;
            #[inline]
            fn $method(self, rhs: Trace<T>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    }
}

operator_impl_value_value!(impl Add for Trace { fn add });
operator_impl_reference_value!(impl Add for Trace { fn add });
operator_impl_value_reference!(impl Add for Trace { fn add });

/**
 * Multiplication for two referenced traces of the same type.
 */
impl <T: Numeric + Primitive> Mul for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn mul(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone() * rhs.number.clone(),
            // u'v + uv'
            derivative: (self.derivative.clone() * rhs.number.clone())
                + (self.number.clone() * rhs.derivative.clone()),
        }
    }
}

operator_impl_value_value!(impl Mul for Trace { fn mul });
operator_impl_reference_value!(impl Mul for Trace { fn mul });
operator_impl_value_reference!(impl Mul for Trace { fn mul });


/**
 * Subtraction for two referenced traces of the same type.
 */
impl <T: Numeric + Primitive> Sub for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn sub(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone() - rhs.number.clone(),
            derivative: self.derivative.clone() - rhs.derivative.clone(),
        }
    }
}

operator_impl_value_value!(impl Sub for Trace { fn sub });
operator_impl_reference_value!(impl Sub for Trace { fn sub });
operator_impl_value_reference!(impl Sub for Trace { fn sub });

/**
 * Division for two referenced traces of the same type.
 */
impl <T: Numeric + Primitive> Div for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn div(self, rhs: &Trace<T>) -> Self::Output {
        Trace {
            number: self.number.clone() / rhs.number.clone(),
            // (u'v - uv') / v^2
            derivative: (
                (
                    (self.derivative.clone() * rhs.number.clone())
                    - (self.number.clone() * rhs.derivative.clone())
                )
                / (rhs.number.clone() * rhs.number.clone())
            ),
        }
    }
}

operator_impl_value_value!(impl Div for Trace { fn div });
operator_impl_reference_value!(impl Div for Trace { fn div });
operator_impl_value_reference!(impl Div for Trace { fn div });

/**
 * Negation for a referenced Trace of some type.
 */
impl <T: Numeric + Primitive> Neg for &Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn neg(self) -> Self::Output {
        Trace::<T>::zero() - self
    }
}

/**
 * Negation for a Trace by value of some type.
 */
impl <T: Numeric + Primitive> Neg for Trace<T>
where for<'a> &'a T: NumericRef<T> {
    type Output = Trace<T>;
    fn neg(self) -> Self::Output {
        Trace::<T>::zero() - self
    }
}


use std::cell::RefCell;

type Index = usize;

/**
 * A list tracking the operations performed in a forward pass.
 */
// TODO: mention that every method involving this could panic if multiple
// mutable borrows are attempted at once
#[derive(Debug)]
pub struct WengertList<T: Primitive> {
    // It is neccessary to wrap the vec in a RefCell to allow for mutating
    // this list from immutable references held by each TODO
    operations: RefCell<Vec<Operation<T>>>
}

#[derive(Debug)]
struct Operation<T: Primitive> {
    left_parent: Index,
    right_parent: Index,
    left_derivative: T,
    right_derivative: T,
}

/**
 * Any operation of a Cloneable type implements clone
 */
impl <T: Clone + Primitive> Clone for Operation<T> {
    fn clone(&self) -> Self {
        Operation {
            left_parent: self.left_parent,
            right_parent: self.right_parent,
            left_derivative: self.left_derivative.clone(),
            right_derivative: self.right_derivative.clone(),
        }
    }
}

#[derive(Debug)]
pub struct Record<'a, T: Primitive> {
    // A record consists of a number used in the forward pass, as
    // well as a WengertList of operations performed on the numbers
    // and each record needs to know which point in the history they
    // are for.
    pub number: T,
    history: Option<&'a WengertList<T>>,
    pub position: Index,
}

impl <'a, T: Numeric + Primitive> Record<'a, T> {
    /*
     * Creates an untracked Record which has no backing WengertList.
     *
     * This is only implemented for satisfying the type level constructors
     * required by Numeric, and otherwise would not be available.
     */
    fn untracked(x: T) -> Record<'a, T> {
        Record {
            number: x,
            history: None,
            position: 0,
        }
    }
}

impl <T: Primitive> WengertList<T> {
    pub fn new() -> WengertList<T> {
        WengertList {
            operations: RefCell::new(Vec::new())
        }
    }
}

impl <T: Numeric + Primitive> WengertList<T> {
    /**
     * Creates a record backed by this WengertList.
     */
    pub fn record<'a>(&'a self, x: T) -> Record<'a, T> {
        Record {
            number: x,
            history: Some(self),
            position: self.append_nullary(),
        }
    }

    /**
     * Adds a value to the list which does not have any parent values.
     */
    fn append_nullary(&self) -> Index {
        let mut operations = self.operations.borrow_mut();
        // insert into end of list
        let index = operations.len();
        operations.push(Operation {
            // this index of the child is used for both indexes as these
            // won't be needed but will always be valid (ie point to a
            // real entry on the list)
            left_parent: index,
            right_parent: index,
            // for the parents 0 is used to zero out these calculations
            // as there are no parents
            left_derivative: T::zero(),
            right_derivative: T::zero(),
        });
        index
    }

    /**
     * Adds a value to the list which has one parent.
     *
     * For an output w_N which depends on one parent w_N-1
     * the derivative cached here is δw_N / δw_N-1
     *
     * For example, if z = sin(x), then δz/δx = cos(x)
     */
    fn append_unary(&self, parent: Index, derivative: T) -> Index {
        let mut operations = self.operations.borrow_mut();
        // insert into end of list
        let index = operations.len();
        operations.push(Operation {
            left_parent: parent,
            // this index of the child is used as this index won't be needed
            // but will always be valid (ie points to a real entry on the list)
            right_parent: index,
            left_derivative: derivative,
            // for the right parent 0 is used to zero out this calculation
            // as there is no right parent
            right_derivative: T::zero(),
        });
        index
    }

    /**
     * Adds a value to the list which has two parents.
     *
     * For an output w_N which depends on two parents w_N-1
     * and w_N-2 the derivatives cached here are δw_N / δw_N-1
     * and δw_N / δw_N-2.
     *
     * For example, if z = y + x, then δz/δy = 1 and δz/δx = 1
     * For example, if z = y * x, then δz/δy = x and δz/δ/x = y
     */
    fn append_binary(&self,
            left_parent: Index, left_derivative: T,
            right_parent: Index, right_derivative: T) -> Index {
        let mut operations = self.operations.borrow_mut();
        // insert into end of list
        let index = operations.len();
        operations.push(Operation {
            left_parent: left_parent,
            right_parent: right_parent,
            left_derivative: left_derivative,
            right_derivative: right_derivative,
        });
        index
    }
}

impl <'a, T: Numeric + Primitive> ZeroOne for Record<'a, T> {
    #[inline]
    fn zero() -> Record<'a, T> {
        Record::untracked(T::zero())
    }
    #[inline]
    fn one() -> Record<'a, T> {
        Record::untracked(T::one())
    }
}

impl <'a, T: Numeric + Primitive> FromUsize for Record<'a, T> {
    #[inline]
    fn from_usize(n: usize) -> Option<Record<'a, T>> {
        Some(Record::untracked(T::from_usize(n)?))
    }
}

/**
 * Any record of a Cloneable type implements clone
 */
impl <'a, T: Clone + Primitive> Clone for Record<'a, T> {
    fn clone(&self) -> Self {
        Record {
            number: self.number.clone(),
            history: self.history.clone(),
            position: self.position.clone(),
        }
    }
}

/**
 * Compares two record's referenced WengertLists.
 *
 * If either Record is missing a reference to a WengertList then
 * this is trivially 'true', in so far as we will use the WengertList of
 * the other one.
 *
 * If both records have a WengertList, then checks that the lists are
 * the same.
 */
fn same_list<'a, 'b, T: Primitive>(a: &Record<'a, T>, b: &Record<'b, T>) -> bool {
    match (a.history, b.history) {
        (None, None) => true,
        (Some(_), None) => true,
        (None, Some(_)) => true,
        (Some(list_a), Some(list_b)) => (
            list_a as *const WengertList<T> == list_b as *const WengertList<T>
        ),
    }
}

impl <'a, T: Numeric + Primitive> Record<'a, T>
where for<'t> &'t T: NumericRef<T> {
    pub fn gradients(&self) -> Vec<T> {
        let history = match self.history {
            None => panic!("Record has no WengertList"),
            Some(h) => h,
        };
        let operations = history.operations.borrow();

        let mut derivatives = vec![ T::zero(); operations.len() ];

        // δy/δy = 1
        derivatives[self.position] = T::one();

        // Go back up the computation graph to the inputs
        for i in (0..operations.len()).rev() {
            let operation = operations[i].clone();
            let derivative = derivatives[i].clone();
            // The chain rule allows breaking up the derivative of the output y
            // with respect to the input x into many smaller derivatives that
            // are summed together.
            // δy/δx = δy/δw * δw/δx
            // δy/δx = sum for all i parents of y ( δy/δw_i * δw_i/δx )
            derivatives[operation.left_parent] =
                derivatives[operation.left_parent].clone()
                + derivative.clone() * operation.left_derivative;
            derivatives[operation.right_parent] =
                derivatives[operation.right_parent].clone()
                + derivative * operation.right_derivative;
        }

        derivatives
    }
}

/**
 * Addition for two records of the same type with both referenced and
 * both using the same WengertList.
 */
impl <'a, T: Numeric + Primitive> Add for &Record<'a, T>
where for<'t> &'t T: NumericRef<T> {
    type Output = Record<'a, T>;
    #[inline]
    fn add(self, rhs: &Record<'a, T>) -> Self::Output {
        assert!(same_list(self, rhs), "Records must be using the same WengertList");
        let history = match (self.history, rhs.history) {
            // If neither inputs have a WengertList then we don't need to record
            // the computation graph at this point because neither are inputs to
            // the overall function.
            // eg f(x, y) = ((1 + 1) * x) + (2 * (1 + y)) needs the records
            // for 2x + (2 * (1 + y)) to be stored, but we don't care about the derivatives
            // for 1 + 1, because neither were inputs to f.
            (None, None) => return Record {
                number: self.number.clone() + rhs.number.clone(),
                history: None,
                position: 0,
            },
            (Some(h), None) => h,
            (None, Some(h)) => h,
            (Some(h), Some(_)) => h,
        };
        Record {
            number: self.number.clone() + rhs.number.clone(),
            history: Some(history),
            position: history.append_binary(
                self.position,
                // δ(self + rhs) / δself = 1
                T::one(),
                rhs.position,
                // δ(self + rhs) / rhs = 1
                T::one()
            ),
        }
    }
}

/**
 * Multiplication for two records of the same type with both referenced and
 * both using the same WengertList.
 */
impl <'a, T: Numeric + Primitive> Mul for &Record<'a, T>
where for<'t> &'t T: NumericRef<T> {
    type Output = Record<'a, T>;
    #[inline]
    fn mul(self, rhs: &Record<'a, T>) -> Self::Output {
        assert!(same_list(self, rhs), "Records must be using the same WengertList");
        let history = match (self.history, rhs.history) {
            (None, None) => return Record {
                number: self.number.clone() * rhs.number.clone(),
                history: None,
                position: 0,
            },
            (Some(h), None) => h,
            (None, Some(h)) => h,
            (Some(h), Some(_)) => h,
        };
        Record {
            number: self.number.clone() * rhs.number.clone(),
            history: Some(history),
            position: history.append_binary(
                self.position,
                // δ(self * rhs) / δself = rhs
                rhs.number.clone(),
                rhs.position,
                // δ(self * rhs) / rhs = self
                self.number.clone()
            ),
        }
    }
}
