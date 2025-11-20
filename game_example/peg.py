import numpy as np
from typing import List, Tuple

from game_example.base_game import BaseGame


class PursuitEvasionGame(BaseGame):
    def __init__(self, occupancy_matrix: np.ndarray, capture_radius: int = 1,
                 pursuer_action_space: List[Tuple[int, int]] = None,
                 evader_action_space: List[Tuple[int, int]] = None,
                 transition_eps: float = 0.0, gamma=0.9):
        """
        Initialize an instance of pursuit-evasion game.
        Player 1 being the pursuer maximizing and player 2 being the evader minimizing.
        :param occupancy_matrix: Representing the grid environment. 1 represents obstacle and 0 represents free space.
        :param capture_radius: Capture radius of the pursuer.
        :param pursuer_action_space: Action space (list of movement vectors) of the pursuer.
        :param evader_action_space: Action space (list of movement vectors) of the evader.
        :param transition_eps: stochastic transition, the probability of staying at the current location
        :param gamma: discount factor.
        """
        print("Initializing Pursuit-Evasion Game...")
        self.occupancy_matrix = occupancy_matrix
        self.transition_eps = transition_eps
        self.capture_radius = capture_radius
        if pursuer_action_space is None:
            pursuer_action_space = [(-1, 0), (1, 0), (0, -1), (0, 1),
                                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                                    (0, 0)]
        self.pursuer_action_space = pursuer_action_space

        if evader_action_space is None:
            evader_action_space = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        self.evader_action_space = evader_action_space

        # generate states and state mappings
        self.n_rows, self.n_cols = occupancy_matrix.shape
        self.n_individual_states = self.n_rows * self.n_rows - np.sum(occupancy_matrix)
        n_states = self.n_individual_states ** 2
        self._n_states = n_states

        self._generate_individual_state_mapping()

        # generate actions and action mappings
        n_actions_1, n_actions_2, self.action_mappings_1, self.action_mappings_2 = self._generate_action_mapping()

        # initialize base game
        super().__init__(n_states=n_states, n_actions_1=n_actions_1, n_actions_2=n_actions_2, gamma=gamma)

        # generate the terminal/capture states
        terminal_states = self._generate_terminal_states()
        self.set_terminal_states(terminal_states)

        # generate rewards
        print("    Generating rewards for PEG...")
        rewards = self._generate_rewards(terminal_states)
        self.set_rewards(rewards)

        # generate compressed transitions
        print("    Generating compressed transitions for PEG...")
        transition_to_states, transition_probs = self._generate_compressed_transitions()
        self.set_compressed_transitions(transition_to_states, transition_probs)

        print("PEG initialized!")

    def _generate_terminal_states(self) -> List[int]:
        """
        Generate terminal/capture states for the game.
        The pursuer captures the evader when the evader is within its capture radius.
        :return: list of terminal states
        """
        terminal_states = []
        for s in range(self.get_n_states()):
            individual_state_1, individual_state_2 = self.game2individual_state(s)
            row_1, col_1 = self.individual_state_mapping[individual_state_1]
            row_2, col_2 = self.individual_state_mapping[individual_state_2]
            if abs(row_1 - row_2) <= self.capture_radius and abs(col_1 - col_2) <= self.capture_radius:
                terminal_states.append(s)
        return terminal_states

    def _generate_rewards(self, terminal_states: List[int]) -> np.ndarray:
        """
        Generate rewards for the game. The states within the terminal states correspond to capture states.
        The pursuer receives +1 for capture. All other states has 0 reward.
        :param terminal_states: List of terminal states
        :return: The state-dependent rewards.
        """
        rewards = np.zeros((self.get_n_states(),))
        for s in terminal_states:
            rewards[s] = 1
        return rewards

    def _generate_compressed_transitions(self) -> Tuple[List[List[List[List[int]]]], List[List[List[List[float]]]]]:
        """
        Generate compressed transitions for the game. The transitions are stored in a list of dictionaries, where each
        dictionary represents the transition probabilities for a given state and action.
        :return: None
        """
        transitions_prob = []
        transitions_to_state = []
        for s in range(self.get_n_states()):
            transitions_prob.append([])
            transitions_to_state.append([])

            individual_state_1, individual_state_2 = self.game2individual_state(s)
            row_1, col_1 = self.individual_state_mapping[individual_state_1]
            row_2, col_2 = self.individual_state_mapping[individual_state_2]

            for a1 in range(self.get_n_action1(s)):
                transitions_prob[s].append([])
                transitions_to_state[s].append([])
                for a2 in range(self.get_n_action2(s)):
                    probs = []
                    next_states = []
                    dr1, dc1 = self.action_mappings_1[s][a1]
                    dr2, dc2 = self.action_mappings_2[s][a2]

                    next_row_1 = row_1 + dr1
                    next_col_1 = col_1 + dc1
                    next_row_2 = row_2 + dr2
                    next_col_2 = col_2 + dc2

                    assert self.occupancy_matrix[next_row_1, next_col_1] == 0 and \
                           self.occupancy_matrix[next_row_2, next_col_2] == 0, "Invalid move, obstacle."
                    assert 0 <= next_row_1 < self.n_rows and 0 <= next_col_1 < self.n_cols and \
                           0 <= next_row_2 < self.n_rows and 0 <= next_col_2 < self.n_cols, \
                        "Invalid move, out of bounds."

                    next_states_1 = [self.pos2individual_state((next_row_1, next_col_1)), individual_state_1]
                    next_states_2 = [self.pos2individual_state((next_row_2, next_col_2)), individual_state_2]
                    prob_1 = [1 - self.transition_eps, self.transition_eps]
                    prob_2 = [1 - self.transition_eps, self.transition_eps]

                    for s1, p1 in zip(next_states_1, prob_1):
                        for s2, p2 in zip(next_states_2, prob_2):
                            next_states.append(self.individual2game_state(s1, s2))
                            probs.append(p1 * p2)
                    transitions_prob[s][a1].append(probs)
                    transitions_to_state[s][a1].append(next_states)

        return transitions_to_state, transitions_prob

    def _generate_individual_state_mapping(self) -> None:
        """
        Generate mapping from individual states to grid positions.
        :return: None
        """
        self.individual_state_mapping = []
        index = 0
        for row in range(self.occupancy_matrix.shape[0]):
            for col in range(self.occupancy_matrix.shape[1]):
                if self.occupancy_matrix[row, col] == 0:
                    self.individual_state_mapping.append((row, col))
                    index += 1

    def _generate_action_mapping(self):
        n_actions_1, n_actions_2 = [], []
        individual_action_mapping_1, individual_action_mapping_2 = [], []
        action_mapping_1, action_mapping_2 = [], []
        for individual_s in range(self.n_individual_states):
            row, col = self.individual_state_mapping[individual_s]
            individual_action_mapping_1.append(self._get_possible_actions(row, col,
                                                                        action_space=self.pursuer_action_space))
            individual_action_mapping_2.append(self._get_possible_actions(row, col,
                                                                        action_space=self.evader_action_space))

        for s in range(self.get_n_states()):
            s1, s2 = self.game2individual_state(s)
            n_actions_1.append(len(individual_action_mapping_1[s1]))
            n_actions_2.append(len(individual_action_mapping_2[s2]))
            action_mapping_1.append(individual_action_mapping_1[s1])
            action_mapping_2.append(individual_action_mapping_2[s2])

        return n_actions_1, n_actions_2, action_mapping_1, action_mapping_2

    def individual_state2pos(self, individual_state: int) -> tuple[int, int]:
        """
        Convert individual state to position in the grid.
        :return: A tuple representing the positions of the individual state in the grid.
        """
        return self.individual_state_mapping[individual_state]

    def pos2individual_state(self, pos: tuple[int, int]) -> int:
        """
        Convert position in the grid to individual state.
        :return: The individual state corresponding to the given position.
        """
        return self.individual_state_mapping.index(pos)

    def individual2game_state(self, individual_state_1: int, individual_state_2: int) -> int:
        """
        Convert individual states to game state.
        :param individual_state_1: individual state of agent 1
        :param individual_state_2: individual state of agent 2
        :return: game/joint state
        """
        return individual_state_1 * self.n_individual_states + individual_state_2

    def game2individual_state(self, game_state: int) -> tuple[int, int]:
        """
        Convert game state to individual states.
        :param game_state: joint state of the two players
        :return: individual states of agent 1 and agent 2
        """
        individual_state_1 = game_state // self.n_individual_states
        individual_state_2 = game_state % self.n_individual_states
        return individual_state_1, individual_state_2

    def game_state2pos(self, game_state: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Convert game state to position in the grid.
        :param game_state: int, joint state
        :return: pos (row, col) of the pursuer and the evader
        """
        state_1, state_2 = self.game2individual_state(game_state)
        pos_1, pos_2 = self.individual_state2pos(state_1), self.individual_state2pos(state_2)
        return pos_1, pos_2

    def pos2game_state(self, pos_1: tuple[int, int], pos_2: tuple[int, int]) -> int:
        """
        Convert positions in the grid to game state.
        :param pos_1: position of the pursuer
        :param pos_2: position of the evader
        :return: joint/game state
        """
        state_1, state_2 = self.pos2individual_state(pos_1), self.pos2individual_state(pos_2)
        game_state = self.individual2game_state(state_1, state_2)
        return game_state

    def _get_possible_actions(self, row, col, action_space: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Get possible actions for the agent at the given position in the grid.
        :param row: Row position of the agent.
        :param col: Column position of the agent.
        :param action_space: List of possible actions for the agent.
        :return: A list of possible actions.
        """
        possible_actions = []
        for dr, dc in action_space:
            row_ = row + dr
            col_ = col + dc
            if 0 <= row_ < self.n_rows and 0 <= col_ < self.n_cols and self.occupancy_matrix[row_, col_] == 0:
                possible_actions.append((dr, dc))
        return possible_actions

if __name__ == '__main__':
    from game_example.solver import NashSolver
    occupancy_matrix = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    peg = PursuitEvasionGame(occupancy_matrix=occupancy_matrix, capture_radius=1, transition_eps=0.0, gamma=0.9)

    solver = NashSolver(game=peg)
    solver.solve(verbose=True, n_policy_eval=5, eps=0.1, n_workers=8)