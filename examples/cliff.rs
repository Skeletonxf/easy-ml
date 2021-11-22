use easy_ml::matrices::Matrix;

use std::fmt;

/**
 * Example 6.6 Cliff Walking from Reinforcement Learning: An Introduction by
 * Richard S. Sutton and Andrew G. Barto
 */

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
enum Cell {
    Path,
    Goal,
    Cliff,
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Cell::Path => "_",
                Cell::Goal => "G",
                Cell::Cliff => "^",
            }
        )
    }
}

fn main() {
    let grid_world = {
        use Cell::Cliff as C;
        use Cell::Goal as G;
        use Cell::Path as P;
        Matrix::from(vec![
            vec![P, P, P, P, P, P, P, P, P, P, P, P],
            vec![P, P, P, P, P, P, P, P, P, P, P, P],
            vec![P, P, P, P, P, P, P, P, P, P, P, P],
            vec![P, C, C, C, C, C, C, C, C, C, C, G],
        ])
    };
    println!("{}", grid_world);
    // TODO: Implement agent, actions, Q-sarsa,
}
