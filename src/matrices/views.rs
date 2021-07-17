/*!
 * Ways to slice into a Matrix, viewing only part of the whole data.
 *
 * Not stable or public API yet
 */

 use std::ops::Range;

 use crate::matrices::{Matrix};
 use crate::matrices::slices::{Slice, Slice2D};

struct MatrixView<T> {
    source: Matrix<T>, // this should be either & or &mut
    rows: Lookup,
    columns: Lookup,
}

enum Lookup {
    Range(Range<usize>),
    Mapping(Vec<usize>),
}

impl Lookup {
    /**
     * Maps from a coordinate space of the ith index accessible by this view to the actual index
     * into the entire matrix data.
     */
    #[inline]
    fn map_index(&self, index: usize) -> Option<usize> {
        match self {
            Lookup::Range(range) => {
                let length = range.end - range.start;
                if index <= length {
                    Some(index + range.start)
                } else {
                    None
                }
            },
            Lookup::Mapping(lookup) => {
                if index < lookup.len() {
                    Some(lookup[index])
                } else {
                    None
                }
            },
        }
    }
}

impl <T> MatrixView<T> {
    fn slice_to_lookup(length: usize, slice: Slice) -> Lookup {
        match slice {
            // Anything which can be expressed as a range of accepted
            // indexes (likely most actual use cases of slices) can be
            // turned into an arithmetic lookup
            Slice::All() => Lookup::Range(0..length),
            Slice::None() => Lookup::Range(0..0),
            Slice::Single(i) => Lookup::Range(i..(i+1)),
            Slice::Range(range) => Lookup::Range(range),
            slice => {
                // For the general case create a lookup table
                let mut lookup = Vec::with_capacity(length);
                for i in 0..length {
                    if slice.accepts(i) {
                        lookup.push(i);
                    }
                }
                Lookup::Mapping(lookup)
            },
        }
    }

    fn from(source: Matrix<T>, slice: Slice2D) -> MatrixView<T> {
        let rows = source.rows();
        let columns = source.columns();
        MatrixView {
            source,
            rows: MatrixView::<T>::slice_to_lookup(rows, slice.rows),
            columns: MatrixView::<T>::slice_to_lookup(columns, slice.columns),
        }
    }
}
