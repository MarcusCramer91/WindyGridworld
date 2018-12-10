import numpy as np
import matplotlib.pyplot as plt


class Gridworld:
    def __init__(self):
        self.world_width = 10
        self.world_height = 7
        self.current_position = (3, 0)
        self.winds = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)
        self.FINAL_POSITION = (3, 7)

    def is_final_position(self):
        if (self.current_position == self.FINAL_POSITION):
            return True

    # actually move the actor in the world
    def move(self, move):
        new_position = list(self.current_position)
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

        # apply winds to the current position
        new_position[0] -= self.winds[self.current_position[1]]

        # ensure that the map cannot be left
        if new_position[0] < 0:
            new_position[0] = 0
        self.current_position = tuple(new_position)

        # determine reward contribution
        if (self.is_final_position()):
            reward = 1
        else:
            reward = -1
        return (self.current_position, reward)

    def get_possible_moves(self, state):
        possible_moves = []
        if state[0] != 0:
            possible_moves.append("up")
        if state[0] != (self.world_height - 1):
            possible_moves.append("down")
        if state[1] != 0:
            possible_moves.append("left")
        if state[1] != (self.world_width - 1):
            possible_moves.append("right")
        return possible_moves

    def get_world_dimensions(self):
        return self.world_height, self.world_width

    def reset(self):
        self.current_position = (3,0)

    def get_possible_states(self):
        states = []
        for height in range(self.world_height):
            for width in range(self.world_width):
                states.append((height, width))
        return states


class Agent:
    def __init__(self, epsilon, alpha, world: Gridworld):
        self.epsilon = epsilon
        self.alpha = alpha
        self.world = world
        self.states = np.full(world.get_world_dimensions(), -1, dtype=float)
        self.states[3,7] = 1
        self.Q = self.initialize_Q(self.world.get_possible_states())

    def initialize_Q(self, states):
        Q = {}
        for s in states:
            Q[s] = {}
            possible_moves = self.world.get_possible_moves(s)
            for a in possible_moves:
                Q[s][a] = 0
        return Q

    def max_Q(self, Q, state):
        max_value = -99999
        max_action = "up"
        for k,v in Q[state].items():
            if v > max_value:
                max_value = v
                max_action = k
        return max_action

    def get_move_epsilon_greedy(self, epsilon, position):
        possible_moves = self.world.get_possible_moves(world.current_position)
        rand = np.random.rand()
        if (rand < epsilon):
            move = possible_moves[np.random.choice(len(possible_moves))]
        else:
            move = self.max_Q(self.Q, position)
        return move

    def sarsa(self, iteration, gamma=0.5, alpha=0.5, epsilon=1):
        visiting = np.zeros(self.world.get_world_dimensions())
        self.world.reset()
        rewards = []
        state_history = []
        visiting = np.zeros(self.world.get_world_dimensions())
        nSteps = 0
        while True:
            nSteps += 1
            current_position = world.current_position[:]
            # move the player
            move = self.get_move_epsilon_greedy(epsilon/iteration, current_position)
            tmp = self.world.move(move)
            visiting[tmp[0][0], tmp[0][1]] += 1
            reward = tmp[1]
            # get the action for the next state
            next_position = world.current_position[:]
            next_move = self.get_move_epsilon_greedy(epsilon/iteration, next_position)

            # update SARSA rule
            self.Q[current_position][move] = self.Q[current_position][move] + alpha * \
                                                            (reward + gamma * self.Q[next_position][next_move] - \
                                                             self.Q[current_position][move])

            rewards.append(reward)
            if self.world.is_final_position():
                break
        return np.sum(rewards)

    def follow_optimal_policy(self):
        self.world.reset()
        rewards = []
        visiting = np.zeros(self.world.get_world_dimensions())
        t = 0
        while True:
            t += 1
            move = self.get_move_epsilon_greedy(epsilon=0, position=self.world.current_position[:])
            tmp = self.world.move(move)
            visiting[tmp[0][0], tmp[0][1]] = t
            rewards.append(tmp[1])
            if self.world.is_final_position():
                break

        reward = np.sum(rewards)
        return reward, visiting

# Logging functionality
def print_world(world: Gridworld):
    dim = world.get_world_dimensions()
    print("----------------------------------------")
    for i in range(dim[0]):
        for j in range(dim[1]):
            if (world.current_position == (i,j)):
                print("X", end = "\t")
            else:
                print("0", end = "\t")
        print()
    print("----------------------------------------")
    print()


# Logging functionality
def print_path(world: Gridworld, visiting):
    dim = world.get_world_dimensions()
    print("Path taken:")
    print("----------------------------------------")
    for i in range(dim[0]):
        for j in range(dim[1]):
            print('{:1.0f}'.format(visiting[(i,j)]), end="\t")
        print()
    print("----------------------------------------")
    print()

if __name__ == '__main__':
    world = Gridworld()
    player = Agent(epsilon=.1, alpha=.2, world=world)
    training_rounds = np.arange(1, 1001, 1)

    rewards = []
    for round in training_rounds:
        rewards.append(player.sarsa(iteration=round, epsilon=1, gamma=0.5, alpha=0.5))

    print("Training done")
    ret = player.follow_optimal_policy()
    print("Reward is", ret[0])

    # print out optimal path chosen
    print_path(player.world,ret[1])

    plt.plot(rewards)
    plt.show()
