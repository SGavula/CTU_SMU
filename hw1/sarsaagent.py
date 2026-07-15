from abstractagent import AbstractAgent
from blackjack import BlackjackEnv, BlackjackObservation, BlackjackAction
from carddeck import *

from collections import defaultdict
import random
import math


class SarsaAgent(AbstractAgent):
    """
    Here you will provide your implementation of SARSA method.
    You are supposed to implement train() method. If you want
    to, you can split the code in two phases - training and
    testing, but it is not a requirement.

    For SARSA explanation check AIMA book or Sutton and Burton
    book. You can choose any strategy and/or step-size function
    (learning rate) as long as you fulfil convergence criteria.
    """
    def __init__(self, env: BlackjackEnv, number_of_episodes: int):
        super().__init__(env, number_of_episodes)
        # Q_table structure: {(state1): [action1, action2], (state2): [action1, action2]} 
        self.Q_table = defaultdict(lambda: [0.0, 0.0]) # This automatically creates [0.0, 0.0] if a state is accessed for the first time
        self.Ns = defaultdict(int)
        self.epsilon = 0
        # self.min_epsilon = 0.0001 # Minimum epsilon
        # self.min_epsilon = 0.05 # Minimum epsilon
        # self.decay_rate = 0.9995
        # self.decay_rate = 0.99999
        self.discount_factor = 0.99

    def get_action(self, state):
        # Using epsilon-greedy strategy for choosing an action
        if random.random() < self.epsilon:
            # Exploration phase: Return random action
            return random.choice([BlackjackAction.HIT, BlackjackAction.STAND]).value
        
        # Exploitation phase: Return action with the highest score for the current state
        q_values = self.Q_table[state]
        optimal_action = np.argmax(q_values)
        return BlackjackAction(optimal_action).value

    def get_state(self, observation):
        player_sum = observation.player_hand.value()
        cards_count = len(observation.player_hand.cards)
        dealers_card = observation.dealer_hand.value()
        has_ace = any(card.rank is Rank.ACE for card in observation.player_hand.cards)
        usable_ace = has_ace and (player_sum + 10 <= 21)
        # First state: (player_sum, dealers_card)
        # First state: (player_sum, dealers_card, usable_ace)
        # First state: (player_sum, cards_count, dealers_card, usable_ace)
        # return (player_sum, dealers_card)
        return (player_sum, dealers_card, usable_ace)
        # return (player_sum, cards_count, dealers_card, usable_ace)

    def train(self):
        # alpha = 0.1
        alpha = 0.01
        printing_treshold = self.number_of_episodes * 0.1
        # TODO your code here (and do not forget to set the number of episodes for learning in main)
        for i in range(self.number_of_episodes):
            if i % printing_treshold == 0:
                print(i)
            observation, _ = self.env.reset()
            # print("Observation: ", observation)
            terminal = False
            
            current_state = self.get_state(observation)
            action = self.get_action(current_state)
            # print("Current state: ", current_state)
            # break

            while not terminal:
                self.Ns[current_state] += 1
                # action = self.get_action(current_state)
                # print("Action: ", action)
                next_observation, reward, terminal, _, _ = self.env.step(action)
                # print("Observation: ", next_observation)

                next_state = self.get_state(next_observation)
                next_action = self.get_action(next_state)

                # Perform SARSA update
                # alpha = 1 / self.Ns[current_state]
                # alpha = 1 / (1 + 0.01 * self.Ns[current_state])

                if terminal:
                    # Final update: no future value
                    target = reward
                else:
                    # Standard update
                    target = reward + self.discount_factor * self.Q_table[next_state][next_action]

                # self.Q_table[current_state][action] = self.Q_table[current_state][action] + alpha * (reward + self.discount_factor * self.Q_table[next_state][next_action] - self.Q_table[current_state][action] )
                self.Q_table[current_state][action] = self.Q_table[current_state][action] + alpha * (target - self.Q_table[current_state][action])

                # Update state
                current_state = next_state
                action = next_action
                # print("Current state: ", current_state)
            # Update epsilon
            # self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            self.epsilon = 1 / math.sqrt(i + 0.001)



    def get_hypothesis(self, observation: BlackjackObservation, terminal: bool, action: int) -> float:
        """
        Implement this method so that I can test your code. This method is supposed to return your learned Q value for
        particular observation and action.

        :param observation: The observation as in the game. Contains information about what the player sees - player's
        hand and dealer's hand.
        :param terminal: Whether the hands were seen after the end of the game, i.e. whether the state is terminal.
        :param action: Action for Q-value.
        :return: The learned Q-value for the given observation and action.
        """
        if terminal:
            return 0.0
    
        state = self.get_state(observation)
        return self.Q_table[state][action]
