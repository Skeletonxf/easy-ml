/*!
# Example of creating an infinite iterator when targeting web assembly

Because Easy ML uses randomness only via the calling code providing a source of random numbers,
you can easily swap out the source of the randomness when targeting more restrictive environments.

Random numbers can be obtained from the JavaScript `Math.random()` method which
already has bindings to Rust provided by the [js-sys](https://crates.io/crates/js-sys) crate.

```
struct EndlessRandomGenerator {}

impl Iterator for EndlessRandomGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        Some(js_sys::Math::random())
    }
}
```
 */
