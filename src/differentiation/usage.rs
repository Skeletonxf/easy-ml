/*!
 * # Usage of Record and Trace
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
 * use easy_ml::numeric::extra::Cos;
 * use easy_ml::numeric::extra::Sin;
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
 * use easy_ml::numeric::extra::{Cos, Sin};
 * // the lifetimes tell the rust compiler that our inputs and outputs
 * // can all live as long as the WengertList
 * fn cartesian<'a>(
 *     r: Record<'a, f32>,
 *     angle: Record<'a, f32>
 * ) -> (Record<'a, f32>, Record<'a, f32>) {
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
 * ## Using Record container
 * ```
 * use easy_ml::differentiation::{Record, RecordTensor, WengertList};
 * use easy_ml::numeric::extra::{Cos, Sin};
 * use easy_ml::tensors::Tensor;
 *
 * // the lifetimes tell the rust compiler that our inputs and outputs
 * // can all live as long as the WengertList
 * fn cartesian<'a>(
 *     r: Record<'a, f32>,
 *     angle: Record<'a, f32>
 * ) -> [Record<'a, f32>; 2] {
 *     let x = r * angle.cos();
 *     let y = r * angle.sin();
 *     [x, y]
 * }
 * // first we must construct a WengertList to create records from
 * let list = WengertList::new();
 * // for this example we also calculate derivatives for 1.5 and 2.5 since you wouldn't use
 * // RecordTensor if you only had a single variable input
 * let R = RecordTensor::variables(&list, Tensor::from([("radius", 2)], vec![ 1.0, 1.5 ]));
 * let A = RecordTensor::variables(&list, Tensor::from([("angle", 2)], vec![ 2.0, 2.5 ]));
 * let (X, Y) = {
 *     let [resultX, resultY] = RecordTensor::from_iters(
 *         [("z", 2)],
 *         R.iter_as_records()
 *             .zip(A.iter_as_records())
 *             // here we operate on each pair of Records as in the prior example, except
 *             // we have to convert to arrays to stream the collection back into RecordTensors
 *             .map(|(r, a)| cartesian(r, a))
 *     );
 *     // we know we can unwrap for this example because we know each iterator contained 2
 *     // elements which matches the shape we're converting back to
 *     (resultX.unwrap(), resultY.unwrap())
 * };
 * // first find dX/dR and dX/dA, we can unwrap because we know X and Y are variables rather than
 * // constants.
 * let X_derivatives = X.derivatives().unwrap();
 * let dX_dR = X_derivatives.map(|d| d.at_tensor(&R));
 * let dX_dA = X_derivatives.map(|d| d.at_tensor(&A));
 * // now find dY/dR and dY/dA
 * let Y_derivatives = Y.derivatives().unwrap();
 * let dY_dR = Y_derivatives.map(|d| d.at_tensor(&R));
 * let dY_dA = Y_derivatives.map(|d| d.at_tensor(&A));
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
 * Storing matrices, tensors or vecs of Records can be inefficienct as it stores the history
 * for each record even when they are the same. Instead, a
 * [RecordTensor](crate::differentiation::RecordTensor) or
 * [RecordMatrix](crate::differentiation::RecordMatrix) can be used,
 * either directly with their elementwise APIs and trait implementations or manipulated as an
 * iterator of Records then collected back into a RecordTensor or RecordMatrix with
 * [RecordContainer::from_iter](crate::differentiation::RecordContainer::from_iter)
 * and [RecordContainer::from_iters](crate::differentiation::RecordContainer::from_iters)
 *
 * ## Substitution
 *
 * There is no need to rewrite the input functions, as you can use the `Numeric` and `Real`
 * traits to write a function that will take floating point numbers, `Trace`s and `Record`s.
 *
 * ```
 * use easy_ml::differentiation::{Trace, Record, WengertList};
 * use crate::easy_ml::numeric::extra::{Real};
 * fn cartesian<T: Real + Copy>(r: T, angle: T) -> (T, T) {
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
 */
