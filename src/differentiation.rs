/*!
 * (Automatic) Differentiation helpers
 *
 * # Automatic Differentiation
 *
 * ## Automatic Differentiation is not [Numerical Differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation)
 *
 * You were probably introduced to differentiation as numeric differentiation,
 * ie if you have a function 3x<sup>2</sup> then you can estimate its gradient
 * at some value x by computing 3x<sup>2</sup> and 3(x+h)<sup>2</sup> where h
 * is a very small value. The tangent line these two points create gives you an approximation
 * of the gradient when you calculate (f(x+h) - f(x)) / h. Unfortunately floating
 * point numbers in computers have limited precision, so this method is only approximate
 * and can result in floating point errors. 1 + 1 might equal 2 but as you go smaller
 * 10<sup>-i</sup> + 10<sup>-i</sup> starts to loook rather like 10<sup>-i</sup> as i goes
 * into double digits.
 *
 * ## Automatic Differentiation is not Symbolic Differentiation
 *
 * If you were taught calculus you have probably done plenty of symbolic differentiation
 * by hand. A function 3x<sup>2</sup> can be symbolically differentiated into 6x by applying
 * simple rules to manipulate the algebra. Unfortunately the rules aren't so simple for
 * more complex expressions such as [exponents](https://www.wolframalpha.com/input/?i=d%28x%5Ee%5E2%29%2Fdx),
 * [logs](https://www.wolframalpha.com/input/?i=d%28log%28log%28x%29%29%29%2Fdx) or
 * [trigonometry](https://www.wolframalpha.com/input/?i=d%28sin%28cos%28x%29%29%29%2Fdx).
 * Symbolic differentiation can give you expressions which are just as or more complicated
 * than the original, and doing it by hand can be error prone. Symbolic Differentiation is
 * also tricky to relate to algorithmic computations that use control structures.
 *
 * ## [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
 *
 * Automatic Differentiation computes the derivative of a function without rewriting
 * the function as symbolic differentiation does and without the precision issues of numerical
 * differentiation by splitting the derivative into lots of tiny steps of basic operations
 * like addition and multiplication. These are combined using the chain rule. The downside
 * is more memory is used than in symbolic or numerical differentiation, as derivatives have
 * to be tracked through the computational graph.
 *
 * # Forward Differentiation
 *
 * Forward Differentiation computes all the gradients in a computational graph with respect
 * to an input. For example, if you have a function f(x, y) = 5x<sup>3</sup> - 4x<sup>2</sup> +
 * 10x - y, then for some actual value of x and y you can compute f(x,y) and δf(x,y)/δx
 * together in one forward pass using forward differentiation. You can also make another pass
 * and compute f(x,y) and δf(x,y)/δy for some actual value of x and y. It is possible to avoid
 * redundantly calculating f(x,y) multiple times, but I am waiting on const generics to implement
 * this. Regardless, forward differentiation requires making at least N+1 passes of the
 * function to compute the derivatives of the output with respect to N inputs - and the current
 * implementation will make 2N. However, you do get the gradients for every output in a
 * single pass. This is poorly suited to neural nets as they often have a single output(loss)
 * to differentiate many many inputs with respect to.
 *
 * # Reverse Mode Differentiation
 *
 * Reverse Mode Differentiation computes all the gradients in a computational graph for
 * the same output. For example, if you have a function f(x, y) = 5x<sup>3</sup> -
 * 4x<sup>2</sup> + 10x - y, then for some actual value of x and y you can compute f(x,y)
 * and store all the intermediate results. You can then run a backward pass on the output
 * of f(x, y) and obtain δf(x,y)/δx and δf(x,y)/δy for the actual values of x and y in a
 * single pass. The catch is that reverse mode must store as many intermediate values as
 * there are steps in the function which can use much more memory than forward mode.
 * Reverse mode also requires making N backward passes to get the gradients for N different
 * outputs. This is well suited to neural nets because we often have a single output (loss)
 * to differentiate many inputs with respect to. However, reverse mode will be slower than
 * forward mode if the number of inputs is small or there are many outputs.
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
 * Importantly this trait is not implemented for Traces (or Records), to stop the compiler
 * from trying to evaluate nested Traces of Traces or Records of Records as Numeric types.
 * There is no reason to create a Trace of a Trace or Record of a Record, it won't do
 * anything a Trace or Record can't except use more memory.
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
 * This is used to perform Forward Automatic Differentiation
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
 *
 * Why the one for the starting derivative? Because δx/δx = 1, as with symbolic
 * differentiation.
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

/**
 * The main set of methods for using Trace types for Forward Differentiation.
 *
 * TODO: explain worked example here
 */
impl <T: Numeric + Primitive> Trace<T> {
    /**
     * Constants are lifted to Traces with a derivative of 0
     *
     * Why zero for the starting derivative? Because for any constant C
     * δC/δx = 0, as with symbolic differentiation.
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
     *
     * Why the one for the starting derivative? Because δx/δx = 1, as with symbolic
     * differentiation.
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
     *
     * In the more general case, if you provide a function with an input x
     * and it returns N outputs y<sub>1</sub> to y<sub>N</sub> then you
     * have computed all the derivatives δy<sub>i</sub>/δx for i = 1 to N.
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
 * A list of operations performed in a forward pass of a dynamic computational graph,
 * used for Reverse Mode Automatic Differentiation.
 *
 * This is dynamic, as in, you build the [Wengert list](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation)
 * at runtime by performing operations like addition and multiplication on
 * Records that were created with a Wengert list.
 *
 * When you perform a backward pass to obtain the gradients you travel back up the
 * computational graph using the stored intermediate values from this list to compute
 * all the gradients of the inputs and every intermediate step with respect to an output.
 *
 * Although sophisticated implementations can make the Wengert list only log(N) in length
 * by storing only some of the intermediate steps of N computational steps, this implementation
 * is not as sophisticated, and will store all of them.
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

// TODO:
// Make proper type for gradients
// Implement other operators and other ref/value combinations
// Make Record implement Numeric
// Add documentation
// Explain seeds for reverse mode
// Stress test reverse mode on matrix / NN setups
// Document panics reverse mode can throw
// Credit Rufflewind for the tutorial
// https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
// https://github.com/Rufflewind/revad/blob/master/src/tape.rs
// Tutorial source code is MIT licensed
// Credit other useful webpages:
// https://medium.com/@marksaroufim/automatic-differentiation-step-by-step-24240f97a6e6
// https://en.m.wikipedia.org/wiki/Automatic_differentiation
// https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/
// The webpage about the power of functional composition will make a good basis for NN
// examples
// https://blog.jle.im/entry/purely-functional-typed-models-1.html
// Leakyness of backprop page will make good intro to NN examples
// https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b

/**
 * A wrapper around a real number which records it going through the computational
 * graph. This is used to perform Reverse Mode Automatic Differentiation.
 */
#[derive(Debug)]
pub struct Record<'a, T: Primitive> {
    // A record consists of a number used in the forward pass, as
    // well as a WengertList of operations performed on the numbers
    // and each record needs to know which point in the history they
    // are for.
    /**
     * The real number
     */
    pub number: T,
    history: Option<&'a WengertList<T>>,
    /**
     * The index of this number in its WengertList.
     *
     * After computing gradients on the WengertList you
     * can index into the gradients to get the derivative
     * for each record.
     */
    pub index: Index,
}

/**
 * The main set of methods for using Record types for Reverse Differentiation.
 *
 * TODO: explain worked example here
 */
impl <'a, T: Numeric + Primitive> Record<'a, T> {
    /**
     * Creates an untracked Record which has no backing WengertList.
     *
     * This is provided for using constants along with records in operations.
     *
     * For example with y = x + 4 the computation graph could be conceived as
     * a y node with parent nodes of x and 4 combined with the operation +.
     * However there is no need to record the derivatives of a constant, so
     * instead the computation graph can be conceived as a y node with a single
     * parent node of x and the unary operation of +4.
     *
     * This is also used for the type level constructors required by Numeric
     * which are also considered constants.
     */
    pub fn constant(c: T) -> Record<'a, T> {
        Record {
            number: c,
            history: None,
            index: 0,
        }
    }

    /**
     * Creates a record backed by the provided WengertList.
     *
     * The record cannot live longer than the WengertList, hence
     * the following example does not compile
     *
     * ```compile_fail
     * use easy_ml::differentiation::Record;
     * use easy_ml::differentiation::WengertList;
     * let record = {
     *     let list = WengertList::new();
     *     Record::variable(1.0, &list)
     * }; // list no longer in scope
     * ```
     *
     * You can alternatively use the [record constructor on the WengertList type](./struct.WengertList.html#method.variable).
     */
    pub fn variable(x: T, history: &'a WengertList<T>) -> Record<'a, T> {
        Record {
            number: x,
            history: Some(history),
            index: history.append_nullary(),
        }
    }
}

impl <'a, T: Numeric + Primitive> Record<'a, T>
where for<'t> &'t T: NumericRef<T> {
    /**
     * Performs a backward pass up this record's WengertList from this
     * record as the output, computing all the derivatives for the inputs
     * involving this output.
     *
     * If you have N inputs x<sub>1</sub> to x<sub>N</sub>, and this output is y,
     * then this computes all the derivatives δy/δx<sub>i</sub> for i = 1 to N.
     */
    pub fn derivatives(&self) -> Vec<T> {
        let history = match self.history {
            None => panic!("Record has no WengertList"),
            Some(h) => h,
        };
        let operations = history.operations.borrow();

        let mut derivatives = vec![ T::zero(); operations.len() ];

        // δy/δy = 1
        derivatives[self.index] = T::one();

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

impl <T: Primitive> WengertList<T> {
    /**
     * Creates a new empty WengertList from which Records can be constructed.
     */
    pub fn new() -> WengertList<T> {
        WengertList {
            operations: RefCell::new(Vec::new())
        }
    }
}

impl <T: Numeric + Primitive> WengertList<T> {
    /**
     * Creates a record backed by this WengertList.
     *
     * You can alternatively use the [record constructor on the Record type](./struct.Record.html#method.variable).
     */
    pub fn variable<'a>(&'a self, x: T) -> Record<'a, T> {
        Record {
            number: x,
            history: Some(self),
            index: self.append_nullary(),
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
        Record::constant(T::zero())
    }
    #[inline]
    fn one() -> Record<'a, T> {
        Record::constant(T::one())
    }
}

impl <'a, T: Numeric + Primitive> FromUsize for Record<'a, T> {
    #[inline]
    fn from_usize(n: usize) -> Option<Record<'a, T>> {
        Some(Record::constant(T::from_usize(n)?))
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
            index: self.index,
        }
    }
}

/**
 * Any record of a Copy type implements Copy
 */
impl <'a, T: Copy + Primitive> Copy for Record<'a, T> { }

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
        match (self.history, rhs.history) {
            // If neither inputs have a WengertList then we don't need to record
            // the computation graph at this point because neither are inputs to
            // the overall function.
            // eg f(x, y) = ((1 + 1) * x) + (2 * (1 + y)) needs the records
            // for 2x + (2 * (1 + y)) to be stored, but we don't care about the derivatives
            // for 1 + 1, because neither were inputs to f.
            (None, None) => Record {
                number: self.number.clone() + rhs.number.clone(),
                history: None,
                index: 0,
            },
            // If only one input has a WengertList treat the other as a constant
            (Some(_), None) => self + &rhs.number,
            (None, Some(_)) => rhs + &self.number,
            (Some(history), Some(_)) => Record {
                number: self.number.clone() + rhs.number.clone(),
                history: Some(history),
                index: history.append_binary(
                    self.index,
                    // δ(self + rhs) / δself = 1
                    T::one(),
                    rhs.index,
                    // δ(self + rhs) / rhs = 1
                    T::one()
                ),
            },
        }
    }
}

impl <'a, T: Numeric + Primitive> Add<&T> for &Record<'a, T>
where for<'t> &'t T: NumericRef<T> {
    type Output = Record<'a, T>;
    #[inline]
    fn add(self, rhs: &T) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone() + rhs.clone(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone() + rhs.clone(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(self + C) / δself = 1
                        T::one()
                    ),
                }
            }
        }
    }
}

macro_rules! record_operator_impl_value_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for two records of the same type.
         */
        impl <'a, T: Numeric + Primitive> $op for Record<'a, T>
        where for<'t> &'t T: NumericRef<T> {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: Record<'a, T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
    }
}

macro_rules! record_operator_impl_value_reference {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
         * Operation for two records of the same type with the right referenced.
         */
        impl <'a, T: Numeric + Primitive> $op<&Record<'a, T>> for Record<'a, T>
        where for<'t> &'t T: NumericRef<T> {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: &Record<'a, T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    }
}

macro_rules! record_operator_impl_reference_value {
    (impl $op:tt for Record { fn $method:ident }) => {
        /**
        * Operation for two records of the same type with the left referenced.
        */
        impl <'a, T: Numeric + Primitive> $op<Record<'a, T>> for &Record<'a, T>
        where for<'t> &'t T: NumericRef<T> {
            type Output = Record<'a, T>;
            #[inline]
            fn $method(self, rhs: Record<'a, T>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    }
}

record_operator_impl_value_value!(impl Add for Record { fn add });
record_operator_impl_reference_value!(impl Add for Record { fn add });
record_operator_impl_value_reference!(impl Add for Record { fn add });

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
        match (self.history, rhs.history) {
            (None, None) => Record {
                number: self.number.clone() * rhs.number.clone(),
                history: None,
                index: 0,
            },
            // If only one input has a WengertList treat the other as a constant
            (Some(_), None) => self * &rhs.number,
            (None, Some(_)) => rhs * &self.number,
            (Some(history), Some(_)) => Record {
                number: self.number.clone() * rhs.number.clone(),
                history: Some(history),
                index: history.append_binary(
                    self.index,
                    // δ(self * rhs) / δself = rhs
                    rhs.number.clone(),
                    rhs.index,
                    // δ(self * rhs) / rhs = self
                    self.number.clone()
                ),
            },
        }
    }
}

impl <'a, T: Numeric + Primitive> Mul<&T> for &Record<'a, T>
where for<'t> &'t T: NumericRef<T> {
    type Output = Record<'a, T>;
    #[inline]
    fn mul(self, rhs: &T) -> Self::Output {
        match self.history {
            None => Record {
                number: self.number.clone() * rhs.clone(),
                history: None,
                index: 0,
            },
            Some(history) => {
                Record {
                    number: self.number.clone() * rhs.clone(),
                    history: Some(history),
                    index: history.append_unary(
                        self.index,
                        // δ(self * C) / δself = C
                        rhs.clone()
                    ),
                }
            }
        }
    }
}

record_operator_impl_value_value!(impl Mul for Record { fn mul });
record_operator_impl_reference_value!(impl Mul for Record { fn mul });
record_operator_impl_value_reference!(impl Mul for Record { fn mul });
