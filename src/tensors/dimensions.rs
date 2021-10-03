#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Dimension {
    name: &'static str,
}

impl Dimension {
    pub fn new(name: &'static str) -> Self {
        Dimension { name }
    }
}

pub fn dimension(name: &'static str) -> Dimension {
    Dimension::new(name)
}

pub fn of(name: &'static str, length: usize) -> (Dimension, usize) {
    (dimension(name), length)
}
