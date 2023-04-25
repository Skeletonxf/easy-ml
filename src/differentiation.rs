#![allow(clippy::double_parens)]
/*!
 * (Automatic) Differentiation helpers
 *
 * # Automatic Differentiation
 *
 * This module provides structs for performing Forward and Reverse Automatic Differentiation
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
 *
 * # Usage
 *
 * Both `Trace` and `Record` for forward and reverse automatic differentiation respectively
 * implement `Numeric` and can generally be treated as normal numbers just like `f32` and `f64`.
 *
 * `Trace` is literally implemented as a dual number, and is more or less a one to one
 * substitution. `Record` requires dynamically building a computational graph of the values
 * and dependencies of each operation performed on them. This means performing operations on
 * records have side effects, they add entries onto a `WengertList`. However, when using
 * `Record` the side effects are abstracted away, just create a `WengertList` before you
 * start creating Records.
 *
 * Given some function from N inputs to M outputs you can pass it `Trace`s or `Record`s
 * and retrieve the first derivative from the outputs for all combinations of N and M.
 * If N >> M then you should use `Record` as reverse mode automatic differentiation is
 * much cheaper. If N << M then you should use `Trace` as it will be much cheaper. If
 * you have large N and M, or small N and M, you might have to benchmark to find which
 * method works best. However, most problems are N > M.
 *
 * For this example we use a function which takes two inputs, r and a, and returns two
 * outputs, x and y.
 *
 * ## Using Trace
 *
 * ```
 * use easy_ml::differentiation::Trace;
 * use crate::easy_ml::numeric::extra::Cos;
 * use crate::easy_ml::numeric::extra::Sin;
 * fn cartesian(r: Trace<f32>, angle: Trace<f32>) -> (Trace<f32>, Trace<f32>) {
 *     let x = r * angle.cos();
 *     let y = r * angle.sin();
 *     (x, y)
 * }
 * // first find dx/dr and dy/dr
 * let (x, y) = cartesian(Trace::variable(1.0), Trace::constant(2.0));
 * let dx_dr = x.derivative;
 * let dy_dr = y.derivative;
 * // now find dx/da and dy/da
 * let (x, y) = cartesian(Trace::constant(1.0), Trace::variable(2.0));
 * let dx_da = x.derivative;
 * let dy_da = y.derivative;
 * ```
 *
 * ## Using Record
 *
 * ```
 * use easy_ml::differentiation::{Record, WengertList};
 * use crate::easy_ml::numeric::extra::{Cos, Sin};
 * // the lifetimes tell the rust compiler that our inputs and outputs
 * // can all live as long as the WengertList
 * fn cartesian<'a>(r: Record<'a, f32>, angle: Record<'a, f32>)
 * -> (Record<'a, f32>, Record<'a, f32>) {
 *     let x = r * angle.cos();
 *     let y = r * angle.sin();
 *     (x, y)
 * }
 * // first we must construct a WengertList to create records from
 * let list = WengertList::new();
 * let r = Record::variable(1.0, &list);
 * let a = Record::variable(2.0, &list);
 * let (x, y) = cartesian(r, a);
 * // first find dx/dr and dx/da
 * let x_derivatives = x.derivatives();
 * let dx_dr = x_derivatives[&r];
 * let dx_da = x_derivatives[&a];
 * // now find dy/dr and dy/da
 * let y_derivatives = y.derivatives();
 * let dy_dr = y_derivatives[&r];
 * let dy_da = y_derivatives[&a];
 * ```
 *
 * ## Differences
 *
 * Notice how in the above examples all the same 4 derivatives are found, but in
 * forward mode we rerun the function with a different input as the sole variable,
 * the rest as constants, whereas in reverse mode we rerun the `derivatives()` function
 * on a different output variable. With Reverse mode we would only pass constants into
 * the `cartesian` function if we didn't want to get their derivatives (and avoid wasting
 * memory on something we didn't need).
 *
 * ## Substitution
 *
 * There is no need to rewrite the input functions, as you can use the `Numeric` and `Real`
 * traits to write a function that will take floating point numbers, `Trace`s and `Record`s.
 *
 * ```
 * use easy_ml::differentiation::{Trace, Record, WengertList};
 * use crate::easy_ml::numeric::Numeric;
 * use crate::easy_ml::numeric::extra::{Real};
 * fn cartesian<T: Numeric + Real + Copy>(r: T, angle: T) -> (T, T) {
 *     let x = r * angle.cos();
 *     let y = r * angle.sin();
 *     (x, y)
 * }
 * let list = WengertList::new();
 * let r_record = Record::variable(1.0, &list);
 * let a_record = Record::variable(2.0, &list);
 * let (x_record, y_record) = cartesian(r_record, a_record);
 * // find dx/dr using reverse mode automatic differentiation
 * let x_derivatives = x_record.derivatives();
 * let dx_dr_reverse = x_derivatives[&r_record];
 * let (x_trace, y_trace) = cartesian(Trace::variable(1.0), Trace::constant(2.0));
 * // now find dx/dr with forward automatic differentiation
 * let dx_dr_forward = x_trace.derivative;
 * assert_eq!(dx_dr_reverse, dx_dr_forward);
 * let (x, y) = cartesian(1.0, 2.0);
 * assert_eq!(x, x_record.number); assert_eq!(x, x_trace.number);
 * assert_eq!(y, y_record.number); assert_eq!(y, y_trace.number);
 * ```
 *
 * ## Equivalance
 *
 * Although in this example the derivatives found are identical, in practise, because
 * forward and reverse mode compute things differently and floating point numbers have
 * limited precision, you should not expect the derivatives to be exactly equal.
 *
 * # Further information
 *
 * - [Automatic Differentiation Step by Step](https://medium.com/@marksaroufim/automatic-differentiation-step-by-step-24240f97a6e6)
 * - [Forward Mode Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers)
 * - [Reverse Mode Automatic Differentiation](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)
 * - [Automatic Differentiation: The most criminally underused tool in the potential machine learning toolbox?](https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/)
 * - [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
 */

pub mod operations;
pub mod record_operations;
pub mod trace_operations;
pub mod container_record;

use crate::numeric::{Numeric, NumericRef};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
 *
 * # Acknowledgments
 *
 * The wikipedia page on [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
 * provided a very useful overview and explanation for understanding Forward Mode Automatic
 * Differentiation as well as the implementation rules.
 */
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Trace<T: Primitive> {
    /**
     * The real number
     */
    pub number: T,
    /**
     * The first order derivative of this number.
     */
    pub derivative: T,
}

/**
 * The main set of methods for using Trace types for Forward Differentiation.
 *
 * The general steps are
 * 1. create one variable
 * 2. create as many constants as needed
 * 3. do operations on the variable and constants
 * 4. the outputs will have derivatives computed which can be accessed from
 * the `.derivative` field, with each derivative being the output with respect
 * to the input variable.
 * 5. if you need derivatives for a different input then do everything all over again
 * or do them all in parallel
 */
impl<T: Numeric + Primitive> Trace<T> {
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
    pub fn derivative(function: impl FnOnce(Trace<T>) -> Trace<T>, x: T) -> T {
        (function(Trace::variable(x))).derivative
    }
}

impl<T: Numeric + Primitive> Trace<T>
where
    for<'a> &'a T: NumericRef<T>,
{
    /**
     * Creates a new Trace from a reference to an existing Trace by applying
     * some unary function to it which operates on the type the Trace wraps.
     *
     * To compute the new trace, the unary function of some input x to some
     * output y is needed along with its derivative with respect to its input x.
     *
     * For example, tanh is a commonly used activation function, but the Real trait
     * does not include this operation and Trace has no operations for it specifically.
     * However, you can use this function to compute the tanh of a Trace like so:
     *
     * ```
     * use easy_ml::differentiation::Trace;
     * let x = Trace::variable(0.7f32);
     * // the derivative of tanh(x) is sech(x) * sech(x) which is equivalent to
     * // 1 / (cosh(x) * cosh(x))
     * let y = x.unary(|x| x.tanh(), |x| 1.0 / (x.cosh() * x.cosh()));
     * assert_eq!(y.derivative, 1.0f32 / (0.7f32.cosh() * 0.7f32.cosh()));
     * ```
     */
    #[inline]
    pub fn unary(&self, fx: impl Fn(T) -> T, dfx_dx: impl Fn(T) -> T) -> Trace<T> {
        Trace {
            number: fx(self.number.clone()),
            derivative: self.derivative.clone() * dfx_dx(self.number.clone()),
        }
    }

    /**
     * Creates a new Trace from a reference to two existing Traces by applying
     * some binary function to them which operates on two arguments of the type
     * the Traces wrap.
     *
     * To compute the new trace, the binary function of some inputs x and y to some
     * output z is needed along with its derivative with respect to its first input x and
     * its derivative with respect to its second input y.
     *
     * For example, atan2 takes two arguments, but the Real trait
     * does not include this operation and Trace has no operations for it specifically.
     * However, you can use this function to compute the atan2 of two Traces like so:
     *
     * ```
     * use easy_ml::differentiation::Trace;
     * let x = Trace::variable(3.0f32);
     * let y = Trace::variable(3.0f32);
     * // the derivative of atan2 with respect to x is y/(x*x + y*y)
     * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdx
     * // the derivative of atan2 with respect to y is -x/(x*x + y*y)
     * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdy
     * let z = x.binary(&y,
     *     |x, y| x.atan2(y),
     *     |x, y| y/((x*x) + (y*y)),
     *     |x, y| -x/((x*x) + (y*y))
     * );
     * ```
     */
    #[inline]
    pub fn binary(
        &self,
        rhs: &Trace<T>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) -> Trace<T> {
        Trace {
            number: fxy(self.number.clone(), rhs.number.clone()),
            #[rustfmt::skip]
            derivative: (
                ((self.derivative.clone() * dfxy_dx(self.number.clone(), rhs.number.clone()))
                + (rhs.derivative.clone() * dfxy_dy(self.number.clone(), rhs.number.clone())))
            ),
        }
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
 * [Records](Record) that were created with that Wengert list.
 *
 * When you perform a backward pass to obtain the gradients you travel back up the
 * computational graph using the stored intermediate values from this list to compute
 * all the gradients of the inputs and every intermediate step with respect to an output.
 *
 * Although sophisticated implementations can make the Wengert list only log(N) in length
 * by storing only some of the intermediate steps of N computational steps, this implementation
 * is not as sophisticated, and will store all of them.
 *
 * # Panics
 *
 * Every operation and nearly every method a Record has involves manipulating the
 * record's history on its referenced WengertList. This WengertList itself maintains
 * a [RefCell](std::cell::RefCell) which tracks
 * borrows at runtime rather than compile time. This is neccessary to maintain the
 * illusion that Records are just ordinary numbers, and the side effects of doing
 * arithmetic with Records are limited to their referenced WengertList. Hence, the Rust
 * compiler correctly infers that it is not safe to share references to WengertLists between
 * threads, nor transfer Records across threads. If you called a method on two Records that both
 * mutably borrowed from the same WengertList at once, which could be trivially done with
 * multiple threads, then the code would panic. Easy ML shouldn't allow you to do this
 * in safe Rust because each mutable borrow of the WengertList is dropped at the end of each
 * Record method call, and you can't call two methods simulatenously without threading.
 */
#[derive(Debug)]
pub struct WengertList<T> {
    // It is neccessary to wrap the vec in a RefCell to allow for mutating
    // this list from immutable references held by each
    operations: RefCell<Vec<Operation<T>>>,
}

/**
 * A binary operation to record on a WengertList. For unary operations the
 * right derivative is set to 0, and for nullary operations both derivatives
 * are set to 0.
 *
 * Each operation acts like a node in an upside down binary tree, with two parents that
 * each node was computed from. The main difference is that the numerical
 * index of those parents in the WengertList is stored, rather than any pointers.
 */
#[derive(Debug)]
struct Operation<T> {
    left_parent: Index,
    right_parent: Index,
    left_derivative: T,
    right_derivative: T,
}

/**
 * Computed derivatives of a computational graph for some output [Record] variable.
 *
 * This can be indexed by any Record used in the computational graph to get
 * the derivative with respect to that input.
 *
 * Indexing using Records not involved in the computational graph, or involved
 * in a different one will return nonsense or index out of bounds and panic. In
 * the future this may be changed to always panic.
 */
#[derive(Debug)]
pub struct Derivatives<T> {
    derivatives: Vec<T>,
}

/**
 * Any derivatives of a Cloneable type implements clone
 */
impl<T: Clone> Clone for Derivatives<T> {
    fn clone(&self) -> Self {
        Derivatives {
            derivatives: self.derivatives.clone(),
        }
    }
}

impl<T: Clone + Primitive> Derivatives<T> {
    /**
     * Quries the derivative at the provided record as input.
     *
     * If you construct a Derivatives object for some output y,
     * and call .at(&x) on it for some input x, this returns dy/dx.
     */
    pub fn at(&self, input: &Record<T>) -> T {
        self.derivatives[input.index].clone()
    }
}

impl<'a, T: Primitive> std::ops::Index<&Record<'a, T>> for Derivatives<T> {
    type Output = T;
    /**
     * Quries the derivative at the provided record as input.
     *
     * If you construct a Derivatives object for some output y,
     * and call .at(&x) on it for some input x, this returns dy/dx.
     */
    fn index(&self, input: &Record<'a, T>) -> &Self::Output {
        &self.derivatives[input.index]
    }
}

impl<T> std::convert::From<Derivatives<T>> for Vec<T> {
    /**
     * Converts the Derivatives struct into a Vec of derivatives that
     * can be indexed with `usize`s. The indexes correspond to the
     * index field on Records.
     */
    fn from(derivatives: Derivatives<T>) -> Self {
        derivatives.derivatives
    }
}

/**
 * Any operation of a Cloneable type implements clone
 */
impl<T: Clone + Primitive> Clone for Operation<T> {
    fn clone(&self) -> Self {
        Operation {
            left_parent: self.left_parent,
            right_parent: self.right_parent,
            left_derivative: self.left_derivative.clone(),
            right_derivative: self.right_derivative.clone(),
        }
    }
}

/**
 * A wrapper around a real number which records it going through the computational
 * graph. This is used to perform Reverse Mode Automatic Differentiation.
 *
 * # Panics
 *
 * Every operation and nearly every method a Record has involves manipulating the
 * record's history on its referenced [WengertList]. This WengertList itself maintains
 * a [RefCell](std::cell::RefCell) which tracks
 * borrows at runtime rather than compile time. This is neccessary to maintain the
 * illusion that Records are just ordinary numbers, and the side effects of doing
 * arithmetic with Records are limited to their referenced WengertList. Hence, the Rust
 * compiler infers that it is not safe to share references to WengertLists between threads,
 * nor transfer Records across threads. If you called a method on two Records that both
 * mutably borrowed from the same WengertList at once, which could be trivially done with
 * multiple threads, then the code would panic. Easy ML shouldn't allow you to do this
 * in safe Rust because each mutable borrow of the WengertList is dropped at the end of each
 * Record method call, and you can't call two methods simulatenously without threading.
 *
 * # Acknowledgments
 *
 * A [tutorial by Rufflewind](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)
 * and the associated [MIT licensed](http://opensource.org/licenses/MIT)
 * [soure code](https://github.com/Rufflewind/revad/blob/master/src/tape.rs) were invaluable
 * in providing understanding on how to implement Reverse Mode Automatic Differentiation.
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
     * The index of this number in its [WengertList]. The first entry will be 0,
     * the next 1 and so on.
     *
     * In normal use cases you should not need to read this field,
     * you can index [Derivatives] directly with Records.
     */
    pub index: Index,
}

/**
 * The main set of methods for using Record types for Reverse Differentiation.
 *
 * The general steps are
 * 1. create a `WengertList`
 * 2. create variables from this list
 * 3. do operations on the variables
 * 4. from the output you want to compute derivatives for call `.derivatives()`
 * 5. index the `Derivatives` object with the index variables to get the derivatives
 * with respect to each input
 * 6. if you want to make another pass call `clear()` on the `WengertList`
 * and then call `reset()` on all of the variables to forget the gradients already
 * computed (the order of `clear` then `reset` is very important!).
 *
 * Constants can be used to save memory if you have numbers that
 * you do not need to compute the gradients with respect to.
 */
impl<'a, T: Numeric + Primitive> Record<'a, T> {
    /**
     * Creates an untracked Record which has no backing WengertList.
     *
     * This is provided for using constants along with Records in operations.
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
     * You can alternatively use the [record constructor on the WengertList type](WengertList::variable()).
     */
    pub fn variable(x: T, history: &'a WengertList<T>) -> Record<'a, T> {
        Record {
            number: x,
            history: Some(history),
            index: history.append_nullary(),
        }
    }

    /**
     * Resets this Record to place it back on its WengertList, for use
     * in performing another derivation after clearing the WengertList.
     */
    pub fn reset(&mut self) {
        match self.history {
            None => (), // noop
            Some(history) => self.index = history.append_nullary(),
        };
    }

    /**
     * A convenience helper function which takes a Record by value and
     * calls [reset](Record::reset()) on it.
     */
    pub fn do_reset(mut x: Record<T>) -> Record<T> {
        x.reset();
        x
    }
}

impl<'a, T: Numeric + Primitive> Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    /**
     * Performs a backward pass up this record's WengertList from this
     * record as the output, computing all the derivatives for the inputs
     * involving this output.
     *
     * If you have N inputs x<sub>1</sub> to x<sub>N</sub>, and this output is y,
     * then this computes all the derivatives δy/δx<sub>i</sub> for i = 1 to N.
     *
     * # Panics
     *
     * Panics if the Record has no backing WengertList, ie it was created as a
     * constant.
     */
    #[track_caller]
    pub fn derivatives(&self) -> Derivatives<T> {
        let history = match self.history {
            None => panic!("Record has no WengertList to find derivatives from"),
            Some(h) => h,
        };
        let operations = history.operations.borrow();

        let mut derivatives = vec![T::zero(); operations.len()];

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
            derivatives[operation.left_parent] = derivatives[operation.left_parent].clone()
                + derivative.clone() * operation.left_derivative;
            derivatives[operation.right_parent] = derivatives[operation.right_parent].clone()
                + derivative * operation.right_derivative;
        }

        Derivatives { derivatives }
    }
}

impl<T: Primitive> WengertList<T> {
    /**
     * Creates a new empty WengertList from which Records can be constructed.
     */
    pub fn new() -> WengertList<T> {
        WengertList {
            operations: RefCell::new(Vec::new()),
        }
    }
}

impl<T: Primitive> Default for WengertList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> WengertList<T> {
    /**
     * Clears a WengertList to make it empty again. After clearing a WengertList
     * you must reset all the Records still using that list. Then you can perform
     * another computation and get new gradients.
     */
    pub fn clear(&self) {
        self.operations.borrow_mut().clear();
    }
}

impl<T: Numeric + Primitive> WengertList<T> {
    /**
     * Creates a record backed by this WengertList.
     *
     * You can alternatively use the [record constructor on the Record type](Record::variable()).
     */
    pub fn variable(&self, x: T) -> Record<T> {
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
     * Adds a number of values to the list which do not have any parent values, returning
     * the index of the first added value, the others will be contiguously afterwards.
     *
     * If values is 0, returns the first index that would be used but wasn't.
     */
    fn append_nullary_repeating(&self, values: usize) -> Index {
        let mut operations = self.operations.borrow_mut();
        // insert into end of list
        let starting_index = operations.len();
        for i in 0..values {
            let index = starting_index + i;
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
        }
        starting_index
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
    fn append_binary(
        &self,
        left_parent: Index,
        left_derivative: T,
        right_parent: Index,
        right_derivative: T,
    ) -> Index {
        let mut operations = self.operations.borrow_mut();
        // insert into end of list
        let index = operations.len();
        operations.push(Operation {
            left_parent,
            right_parent,
            left_derivative,
            right_derivative,
        });
        index
    }
}

/**
 * Any Wengert list of a Cloneable type implements clone
 */
impl<T: Clone + Primitive> Clone for WengertList<T> {
    fn clone(&self) -> Self {
        WengertList {
            operations: RefCell::new(self.operations.borrow().clone()),
        }
    }
}

impl<'a, T: Numeric + Primitive> Record<'a, T>
where
    for<'t> &'t T: NumericRef<T>,
{
    /**
     * Creates a new Record from a reference to an existing Record by applying
     * some unary function to it which operates on the type the Record wraps.
     *
     * To compute the new record, the unary function of some input x to some
     * output y is needed along with its derivative with respect to its input x.
     *
     * For example, tanh is a commonly used activation function, but the Real trait
     * does not include this operation and Record has no operations for it specifically.
     * However, you can use this function to compute the tanh of a Record like so:
     *
     * ```
     * use easy_ml::differentiation::{Record, WengertList};
     * let list = WengertList::new();
     * let x = Record::variable(0.7f32, &list);
     * // the derivative of tanh(x) is sech(x) * sech(x) which is equivalent to
     * // 1 / (cosh(x) * cosh(x))
     * let y = x.unary(|x| x.tanh(), |x| 1.0 / (x.cosh() * x.cosh()));
     * assert_eq!(y.derivatives()[&x], 1.0f32 / (0.7f32.cosh() * 0.7f32.cosh()));
     * ```
     */
    #[inline]
    pub fn unary(&self, fx: impl Fn(T) -> T, dfx_dx: impl Fn(T) -> T) -> Record<T> {
        match self.history {
            None => Record {
                number: fx(self.number.clone()),
                history: None,
                index: 0,
            },
            Some(history) => Record {
                number: fx(self.number.clone()),
                history: Some(history),
                index: history.append_unary(self.index, dfx_dx(self.number.clone())),
            },
        }
    }

    /**
     * Creates a new Record from a reference to two existing Records by applying
     * some binary function to them which operates on two arguments of the type
     * the Records wrap.
     *
     * To compute the new record, the binary function of some inputs x and y to some
     * output z is needed along with its derivative with respect to its first input x and
     * its derivative with respect to its second input y.
     *
     * For example, atan2 takes two arguments, but the Real trait
     * does not include this operation and Record has no operations for it specifically.
     * However, you can use this function to compute the atan2 of two Records like so:
     *
     * ```
     * use easy_ml::differentiation::{Record, WengertList};
     * let list = WengertList::new();
     * let x = Record::variable(3.0f32, &list);
     * let y = Record::variable(3.0f32, &list);
     * // the derivative of atan2 with respect to x is y/(x*x + y*y)
     * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdx
     * // the derivative of atan2 with respect to y is -x/(x*x + y*y)
     * // https://www.wolframalpha.com/input/?i=d%28atan2%28x%2Cy%29%29%2Fdy
     * let z = x.binary(&y,
     *     |x, y| x.atan2(y),
     *     |x, y| y/((x*x) + (y*y)),
     *     |x, y| -x/((x*x) + (y*y))
     * );
     * let derivatives = z.derivatives();
     * let dx = derivatives[&x];
     * let dy = derivatives[&y];
     * ```
     */
    #[inline]
    #[track_caller]
    pub fn binary(
        &self,
        rhs: &Record<'a, T>,
        fxy: impl Fn(T, T) -> T,
        dfxy_dx: impl Fn(T, T) -> T,
        dfxy_dy: impl Fn(T, T) -> T,
    ) -> Record<T> {
        assert!(
            record_operations::same_list(self, rhs),
            "Records must be using the same WengertList"
        );
        match (self.history, rhs.history) {
            (None, None) => Record {
                number: fxy(self.number.clone(), rhs.number.clone()),
                history: None,
                index: 0,
            },
            (Some(history), None) => Record {
                number: fxy(self.number.clone(), rhs.number.clone()),
                history: Some(history),
                index: history.append_unary(
                    // if rhs didn't have a history, don't track that derivative
                    self.index,
                    dfxy_dx(self.number.clone(), rhs.number.clone()),
                ),
            },
            (None, Some(history)) => Record {
                number: fxy(self.number.clone(), rhs.number.clone()),
                history: Some(history),
                index: history.append_unary(
                    // if self didn't have a history, don't track that derivative
                    rhs.index,
                    dfxy_dy(self.number.clone(), rhs.number.clone()),
                ),
            },
            (Some(history), Some(_)) => Record {
                number: fxy(self.number.clone(), rhs.number.clone()),
                history: Some(history),
                index: history.append_binary(
                    self.index,
                    dfxy_dx(self.number.clone(), rhs.number.clone()),
                    rhs.index,
                    dfxy_dy(self.number.clone(), rhs.number.clone()),
                ),
            },
        }
    }
}

#[cfg(test)]
#[should_panic]
#[test]
fn test_record_derivatives_when_no_history() {
    let record = Record::constant(1.0);
    record.derivatives();
}

#[test]
fn test_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<Trace<f64>>();
}

#[test]
fn test_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Trace<f64>>();
}
