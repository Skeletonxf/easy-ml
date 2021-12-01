#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use easy_ml::matrices::Matrix;
use rand::{Rng, SeedableRng};

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

const DIRECTIONS: usize = 4;

impl Direction {
    fn order(&self) -> usize {
        match self {
            Direction::North => 0,
            Direction::East => 1,
            Direction::South => 2,
            Direction::West => 3,
        }
    }

    fn actions() -> [Direction; DIRECTIONS]  {
        [ Direction::North, Direction::East, Direction::South, Direction::West ]
    }
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
    expected_rewards: Vec<f64>,
    steps: u64,
    reward: f64,
}

trait Policy {
    fn choose(&mut self, choices: &[(Direction, f64); DIRECTIONS]) -> Direction;
}

struct EpsilonGreedy {
    rng: rand_chacha::ChaCha8Rng,
    exploration_rate: f64,
}

struct Greedy;

impl Policy for Greedy {
    fn choose(&mut self, choices: &[(Direction, f64); DIRECTIONS]) -> Direction {
        let mut best_q = -f64::INFINITY;
        let mut best_direction = Direction::North;
        for &(d, q) in choices {
            if q > best_q {
                best_direction = d;
                best_q = q;
            }
        }
        best_direction
    }
}

impl Policy for EpsilonGreedy {
    fn choose(&mut self, choices: &[(Direction, f64); DIRECTIONS]) -> Direction {
        let random: f64 = self.rng.gen();
        if random < self.exploration_rate {
            choices[self.rng.gen_range(0..choices.len())].0
        } else {
            Greedy.choose(choices)
        }
    }
}

impl <P: Policy> Policy for &mut P {
    fn choose(&mut self, choices: &[(Direction, f64); DIRECTIONS]) -> Direction {
        P::choose(self, choices)
    }
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
            Direction::East => (std::cmp::min(x1.saturating_add(1), self.tiles.columns() - 1), y1),
            Direction::South => (x1, std::cmp::min(y1.saturating_add(1), self.tiles.rows() - 1)),
            Direction::West => (x1.saturating_sub(1), y1),
        };
        if x1 == x2 && y1 == y2 {
            None
        } else {
            Some((x2, y2))
        }
    }

    fn take_action(&mut self, direction: Direction) -> f64 {
        if let Some((x, y)) = self.step(self.agent, direction) {
            self.agent = (x, y);
            match self.tiles.get(y, x) {
                Cell::Cliff => {
                    self.agent = (0, 3);
                    -100.0
                },
                Cell::Path => -1.0,
                Cell::Goal => 0.0,
            }
        } else {
            -1.0
        }
    }

    // TODO: Generalise to qsarsa
    fn sarsa(&mut self, step_size: f64, mut policy: impl Policy, discount_factor: f64) {
        let (α, γ) = (step_size, discount_factor);
        let mut state = self.agent;
        let mut action = policy.choose(
            &Direction::actions().map(|d| (d, self.q(state, d)))
        );
        while self.tiles.get(self.agent.1, self.agent.0) != Cell::Goal {
            let reward = self.take_action(action);
            self.reward += reward;
            let new_state = self.agent;
            let new_action = policy.choose(
                &Direction::actions().map(|d| (d, self.q(new_state, d)))
            );
            *self.q_mut(state, action) = self.q(state, action) +
                α * (reward + (γ * self.q(new_state, new_action)) - self.q(state, action));
            state = new_state;
            action = new_action;
            self.steps += 1;
        }
    }

    /// Returns the Q value of a particular state action
    fn q(&self, position: Position, direction: Direction) -> f64 {
        *&self.expected_rewards[index(position, direction, self.tiles.columns(), DIRECTIONS)]
    }

    /// Returns a mutable reference to the Q value of a particular state action
    fn q_mut(&mut self, position: Position, direction: Direction) -> &mut f64  {
        &mut self.expected_rewards[index(position, direction, self.tiles.columns(), DIRECTIONS)]
    }
}

// Flattens a 3d position and direction into a 1-dimensional index
fn index(position: Position, direction: Direction, width: usize, directions: usize) -> usize {
    direction.order() + (directions * (position.0 + (position.1 * width)))
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
        agent: (0, 3),
        // Initial values may be arbitary apart from all state - actions on the Goal state
        expected_rewards: vec![ 0.0; DIRECTIONS * 4 * 12 ],
        steps: 0,
        reward: 0.0,
    };
    let episodes = 100;
    let mut policy = EpsilonGreedy {
        rng: rand_chacha::ChaCha8Rng::seed_from_u64(16),
        exploration_rate: 0.1,
    };
    let mut total_steps = 0;
    for n in 0..episodes {
        grid_world.steps = 0;
        grid_world.reward = 0.0;
        grid_world.agent = (0, 3);
        grid_world.sarsa(0.5, &mut policy, 0.9);
        total_steps += grid_world.steps;
        println!("Steps to complete episode {:?}:\t{:?}/{:?}\t\tSum of rewards during episode: {:?}", n, grid_world.steps, total_steps, grid_world.reward);
    }
}
