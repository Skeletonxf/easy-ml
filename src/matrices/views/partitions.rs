use crate::matrices::{Row, Column};

pub struct MatrixPart<'source, T> {
    data: Vec<&'source mut [T]>,
    rows: Row,
    columns: Column,
}

impl <'a, T> MatrixPart<'a, T> {
    pub(crate) fn new(data: Vec<&'a mut [T]>, rows: Row, columns: Column) -> MatrixPart<'a, T> {
        MatrixPart {
            data,
            rows,
            columns
        }
    }
}
