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

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
enum Direction {
    North, East, South, West
}

impl Cell {
    fn to_str(&self) -> &'static str {
        match self {
            Cell::Path => "_",
            Cell::Goal => "G",
            Cell::Cliff => "^",
        }
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.to_str()
        )
    }
}

type Position = (usize, usize);

struct GridWorld {
    tiles: Matrix<Cell>,
    agent: Position,
}

impl fmt::Display for GridWorld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.tiles.row_major_iter().with_index().try_for_each(|((r, c), cell)| {
            write!(
                f,
                "{}{}",
                if (c, r) == self.agent { "A" } else { cell.to_str() },
                if c == self.tiles.columns() - 1 { "\n" } else { ", " }
            )
        })?;
        Ok(())
    }
}

impl GridWorld {
    /// Return the position from moving in this direction. Attempting to go off the grid returns
    /// None
    fn step(&self, current: Position, direction: Direction) -> Option<Position> {
        let (x1, y1) = current;
        let (x2, y2) = match direction {
            Direction::North => (x1, y1.saturating_sub(1)),
            Direction::East => (std::cmp::min(x1.saturating_add(1), self.tiles.columns()), y1),
            Direction::South => (x1, std::cmp::min(y1.saturating_add(1), self.tiles.rows())),
            Direction::West => (x1.saturating_sub(1), y1),
        };
        if x1 == x2 && y1 == y2 {
            None
        } else {
            Some((x2, y2))
        }
    }

    fn take_action(&mut self, direction: Direction) {
        if let Some((x, y)) = self.step(self.agent, direction) {
            self.agent = (x, y);
            if self.tiles.get(y, x) == Cell::Cliff {
                self.agent = (0, 3);
            }
        }
    }
}

fn main() {
    let mut grid_world = GridWorld {
        tiles: {
            use Cell::Cliff as C;
            use Cell::Goal as G;
            use Cell::Path as P;
            Matrix::from(vec![
                vec![P, P, P, P, P, P, P, P, P, P, P, P],
                vec![P, P, P, P, P, P, P, P, P, P, P, P],
                vec![P, P, P, P, P, P, P, P, P, P, P, P],
                vec![P, C, C, C, C, C, C, C, C, C, C, G],
            ])
        },
        agent: (0, 3)
    };
    println!("{}", grid_world);
    grid_world.take_action(Direction::North);
    println!("{}", grid_world);
    // TODO: Implement rewards, Q-sarsa,
}
