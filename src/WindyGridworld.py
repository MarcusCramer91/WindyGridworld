import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class Gridworld:
    def __init__(self):
        self.world_width = 10
        self.world_height = 7
        self.current_position = [3, 0]
        self.winds = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)
        self.FINAL_POSITION = [3, 7]

    def is_final_position(self):
        return self.current_position == self.FINAL_POSITION

    # actually move the actor in the world
    def move(self, move):
        if move == "up":
            self.current_position[0] = self.current_position[0] - 1
        elif move == "down":
            self.current_position[0] = self.current_position[0] + 1
        elif move == "left":
            self.current_position[1] = self.current_position[1] - 1
        elif move == "right":
            self.current_position[1] = self.current_position[1] + 1
        else:
            raise Exception("Illegal move command.")
        self.current_position[0] -= self.winds[self.current_position[1]]

        if (self.current_position[0] < 0):
            self.current_position[0] = 0
        if (self.is_final_position()):
            reward = 1
        else:
            reward = -1
        return (self.current_position, reward)

    # get consequences of actions
    def project_move(self, move):
        new_position = self.current_position[:]
        if move == "up":
            new_position[0] = self.current_position[0] - 1
        elif move == "down":
            new_position[0] = self.current_position[0] + 1
        elif move == "left":
            new_position[1] = self.current_position[1] - 1
        elif move == "right":
            new_position[1] = self.current_position[1] + 1
        else:
            raise Exception("Illegal move command.")
        new_position[0] -= self.winds[new_position[1]]
        if new_position[0] < 0:
            new_position[0] = 0
        return new_position

    def get_possible_moves(self):
        possible_moves = []
        if self.current_position[0] != 0:
            possible_moves.append("up")
        if self.current_position[0] != (self.world_height - 1):
            possible_moves.append("down")
        if self.current_position[1] != 0:
            possible_moves.append("left")
        if self.current_position[1] != (self.world_width - 1):
            possible_moves.append("right")
        return possible_moves

    def get_world_dimensions(self):
        return (self.world_height, self.world_width)

    def reset(self):
        self.current_position = [3,0]


class Agent:
    def __init__(self, epsilon, alpha, world: Gridworld):
        self.epsilon = epsilon
        self.alpha = alpha
        self.world = world
        self.states = np.full(world.get_world_dimensions(), -1, dtype=float)
        self.states[3,7] = 1
        self.Q = defaultdict(lambda: np.zeros(4))


    def epsilon_greedy_move(self, round, epsilon, verbose = False):
        rand = np.random.rand()
        possible_moves = self.world.get_possible_moves()
        # discount epsilon as time passes
        if (rand < epsilon):
            rand = np.random.choice(len(possible_moves))
            move = possible_moves[rand]
        else:
            move = self.get_move_to_best_state(possible_moves)
        if verbose:
            print(move)
        return self.world.move(move)

    def get_move_to_best_state(self, possible_moves):
        move_values = []
        best_move = "right"
        for move in possible_moves:
            new_state = self.world.project_move(move)
            move_values.append(self.states[new_state[0], new_state[1]])

        # if all state values are equal, pick a random move
        if (all(x == move_values[0] for x in move_values)):
            return np.random.choice(possible_moves)
        else:
            return possible_moves[move_values.index(max(move_values))]

    def play_one_round(self, round, verbose = False):
        world.reset()
        rewards = []
        state_history = []
        visiting = np.zeros(world.get_world_dimensions())
        if verbose:
            print_board(world, (3,0))
        nSteps = 0
        while True:
            nSteps += 1
            res = self.epsilon_greedy_move(round, self.epsilon, verbose)
            rewards.append(res[1])
            state_history.append(res[0][:])
            visiting[res[0][0], res[0][1]] += 1
            if self.world.is_final_position():
                break
        self.learn(state_history, rewards, nSteps)
        return np.sum(rewards)

    def follow_optimal_policy(self):
        self.world.reset()
        state_history = []
        rewards = []
        while True:
            res = self.epsilon_greedy_move(round, epsilon = 0, verbose = False)
            state_history.append(res[0][:])
            rewards.append(res[1])
            if self.world.is_final_position():
                break
        res = np.zeros(world.get_world_dimensions())
        for i in range(len(state_history)):
            res[state_history[i][0], state_history[i][1]] = i + 1
        print(res)
        return np.sum(rewards)

    def learn(self, state_history, rewards, nSteps):
        rev = reversed(rewards)
        update = list(reversed(np.cumsum(list(reversed(rewards)))))
        n = np.arange(nSteps, 1, step = -1)
        for i in np.arange(0, len(update)-1):
            update[i] = update[i] / n[i]
        index = 0
        for state in state_history:
            self.states[state[0], state[1]] = (1-self.alpha) * self.states[state[0], state[1]] + self.alpha * update[index]
            index += 1


def print_board(world: Gridworld, position):
    print()
    board = np.zeros(world.get_world_dimensions())
    board[position[0], position[1]] = 1
    print(board)

if __name__ == '__main__':
    all_rewards = []
    world = Gridworld()
    player = Agent(epsilon=.1, alpha=.2, world=world)
    training_rounds = np.arange(1, 1001, 1)

    rewards = []
    for round in training_rounds:
        print(round)
        rewards.append(player.play_one_round(round=round, verbose=False))

    states_printout = player.states[:]
    states_printout = np.around(states_printout, decimals=2)
    print(states_printout)


    ret = player.follow_optimal_policy()
    print("Reward is", ret)

    plt.plot(rewards)
    plt.show()

