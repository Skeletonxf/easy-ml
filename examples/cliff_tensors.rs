#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use easy_ml::tensors::Tensor;
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
    North,
    East,
    South,
    West,
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

    fn actions() -> [Direction; DIRECTIONS] {
        [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ]
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
        write!(f, "{}", self.to_str())
    }
}

type Position = (usize, usize);

struct GridWorld {
    start: Position,
    tiles: Tensor<Cell, 2>,
    agent: Position,
    expected_rewards: Vec<f64>,
    steps: u64,
    reward: f64,
}

trait Policy {
    fn choose(&mut self, choices: &[(Direction, f64); DIRECTIONS]) -> Direction;
}

/// The greedy policy always chooses the action with the best Q value. As the sole policy for
/// a Reinforcement Learning agent, this carries the risk of exploiting the 'best' action the
/// agent finds before it has a chance to find better ones, which can leave an agent stuck in a
/// local optima.
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

/// The ε-greedy policy chooses the greedy action with probability 1-ε, and explores, choosing a
/// random action, with probability ε. This randomness ensures an agent following ε-greedy explores
/// its environment, to stop it getting stuck trying to exploit an (unknown to it) suboptimal
/// greedy choice. However it also makes the agent unable to act 100% optimally since even after
/// learning the true optimal strategy, the exploration rate remains.
struct EpsilonGreedy {
    rng: rand_chacha::ChaCha8Rng,
    exploration_rate: f64,
}

impl Policy for EpsilonGreedy {
    fn choose(&mut self, choices: &[(Direction, f64); DIRECTIONS]) -> Direction {
        let random: f64 = self.rng.r#gen();
        if random < self.exploration_rate {
            choices[self.rng.gen_range(0..choices.len())].0
        } else {
            Greedy.choose(choices)
        }
    }
}

impl<P: Policy> Policy for &mut P {
    fn choose(&mut self, choices: &[(Direction, f64); DIRECTIONS]) -> Direction {
        P::choose(self, choices)
    }
}

impl fmt::Display for GridWorld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let columns = self.tiles.length_of("x").expect("x dimension should exist");
        #[rustfmt::skip]
        self.tiles.iter().with_index().try_for_each(|([r, c], cell)| {
            write!(
                f,
                "{}{}",
                if (c, r) == self.agent { "A" } else { cell.to_str() },
                if c == columns - 1 { "\n" } else { ", " }
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
        #[rustfmt::skip]
        let (x2, y2) = match direction {
            Direction::North => (
                x1,
                y1.saturating_sub(1)
            ),
            Direction::East => (
                std::cmp::min(
                    x1.saturating_add(1),
                    self.tiles.last_index_of("x").expect("x dimension should exist")
                ),
                y1,
            ),
            Direction::South => (
                x1,
                std::cmp::min(
                    y1.saturating_add(1),
                    self.tiles.last_index_of("y").expect("y dimension should exist")
                ),
            ),
            Direction::West => (
                x1.saturating_sub(1),
                y1
            ),
        };
        if x1 == x2 && y1 == y2 {
            None
        } else {
            Some((x2, y2))
        }
    }

    /// Moves the agent in a direction, taking that action, and returns the collected reward.
    fn take_action(&mut self, direction: Direction) -> f64 {
        if let Some((x, y)) = self.step(self.agent, direction) {
            self.agent = (x, y);
            match self.tiles.index_by(["x", "y"]).get([x, y]) {
                Cell::Cliff => {
                    self.agent = self.start;
                    -100.0
                }
                Cell::Path => -1.0,
                Cell::Goal => 0.0,
            }
        } else {
            -1.0
        }
    }

    /// Q-SARSA.
    /// For a step size in the range 0..=1, some policy that chooses an action based on the Q
    /// values learnt so far for that state, a discount factor in the range 0..=1, and a Q-SARSA
    /// factor in the range 0..=1. 0 Q-SARSA is Q-learning, 1 Q-SARSA is SARSA, values in between
    /// update Q values by a factor of both algorithms.
    fn q_sarsa(
        &mut self,
        step_size: f64,
        mut policy: impl Policy,
        discount_factor: f64,
        q_sarsa: f64,
    ) {
        let (α, γ) = (step_size, discount_factor);
        let actions = Direction::actions();
        let mut state = self.agent;
        let mut action = policy.choose(&actions.map(|d| (d, self.q(state, d))));
        while self
            .tiles
            .index_by(["x", "y"])
            .get([self.agent.0, self.agent.1])
            != Cell::Goal
        {
            let reward = self.take_action(action);
            self.reward += reward;
            let new_state = self.agent;
            let new_action = policy.choose(&actions.map(|d| (d, self.q(new_state, d))));
            let greedy_action = Greedy.choose(&actions.map(|d| (d, self.q(new_state, d))));
            let expected_q_value = q_sarsa * self.q(new_state, new_action)
                + ((1.0 - q_sarsa) * self.q(new_state, greedy_action));
            *self.q_mut(state, action) = self.q(state, action)
                + α * (reward + (γ * expected_q_value) - self.q(state, action));
            state = new_state;
            action = new_action;
            self.steps += 1;
        }
    }

    /// Q-learning learns the optimal policy which travels directly along the cliff
    /// to the goal, even though during training the exploration rate occasionally causes
    /// the agent to fall off the cliff.
    #[allow(dead_code)]
    fn q_learning(&mut self, step_size: f64, policy: impl Policy, discount_factor: f64) {
        self.q_sarsa(step_size, policy, discount_factor, 0.0);
    }

    /// SARSA learns the safer route away from the cliff because it factors in the exploration
    /// rate that makes traveling directly along the cliff have a worse performance due to
    /// occasional exploration.
    ///
    /// As the exploration rate tends to 0, SARSA and Q-Learning converge.
    fn sarsa(&mut self, step_size: f64, policy: impl Policy, discount_factor: f64) {
        self.q_sarsa(step_size, policy, discount_factor, 1.0);
    }

    /// Returns the Q value of a particular state action
    fn q(&self, position: Position, direction: Direction) -> f64 {
        let width = self.tiles.length_of("x").expect("x dimension should exist");
        *&self.expected_rewards[index(position, direction, width, DIRECTIONS)]
    }

    /// Returns a mutable reference to the Q value of a particular state action
    fn q_mut(&mut self, position: Position, direction: Direction) -> &mut f64 {
        let width = self.tiles.length_of("x").expect("x dimension should exist");
        &mut self.expected_rewards[index(position, direction, width, DIRECTIONS)]
    }

    fn reset(&mut self) {
        self.steps = 0;
        self.reward = 0.0;
        self.agent = self.start;
    }
}

// Flattens a 3d position and direction into a 1-dimensional index
fn index(position: Position, direction: Direction, width: usize, directions: usize) -> usize {
    direction.order() + (directions * (position.0 + (position.1 * width)))
}

fn main() {
    #[rustfmt::skip]
    let mut grid_world = GridWorld {
        tiles: {
            use Cell::Cliff as C;
            use Cell::Goal as G;
            use Cell::Path as P;
            Tensor::from([("y", 4), ("x", 12)], vec![
                P, P, P, P, P, P, P, P, P, P, P, P,
                P, P, P, P, P, P, P, P, P, P, P, P,
                P, P, P, P, P, P, P, P, P, P, P, P,
                P, C, C, C, C, C, C, C, C, C, C, G,
            ])
        },
        start: (0, 3),
        agent: (0, 3),
        // Initial values may be arbitary apart from all state - actions on the Goal state
        expected_rewards: vec![0.0; DIRECTIONS * 4 * 12],
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
        grid_world.reset();
        grid_world.sarsa(0.5, &mut policy, 0.9);
        total_steps += grid_world.steps;
        println!(
            "Steps to complete episode {:?}:\t{:?}/{:?}\t\tSum of rewards during episode: {:?}",
            n, grid_world.steps, total_steps, grid_world.reward
        );
    }
}
