use std::error::Error;
use std::fmt;

/**
 * An error indicating failure to convert a matrix to a scalar because it is not a unit matrix.
 */
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ScalarConversionError;

impl Error for ScalarConversionError {}

impl fmt::Display for ScalarConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix cannot be converted to a scalar because it is not 1x1")
    }
}

#[test]
fn test_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<ScalarConversionError>();
}

#[test]
fn test_send() {
    fn assert_send<T: Send>() {}
    assert_send::<ScalarConversionError>();
}
