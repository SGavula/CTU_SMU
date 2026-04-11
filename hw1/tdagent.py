from abstractagent import AbstractAgent
from blackjack import BlackjackObservation, BlackjackEnv, BlackjackAction
from carddeck import *


class TDAgent(AbstractAgent):
    """
    Implementation of an agent that plays the same strategy as the dealer.
    This means that the agent draws a card when sum of cards in his hand
    is less than 17.

    Your goal is to modify train() method to learn the state utility function
    and the get_hypothesis() method that returns the state utility function.
    I.e. you need to change this agent to a passive reinforcement learning
    agent that learns utility estimates using temporal difference method.
    """

    def __init__(self, env: BlackjackEnv, number_of_episodes: int):
        super().__init__(env, number_of_episodes)
        self.U = {}
        # Ns dictionary where we save how many times we visited state, for each state
        self.Ns = {}
        self.discount_factor = 0.99

    def get_state(self, observation):
        player_sum = observation.player_hand.value()
        dealers_card = observation.dealer_hand.value()
        return (player_sum, dealers_card)

    def train(self):
        for i in range(self.number_of_episodes):
            observation, _ = self.env.reset()
            terminal = False
            reward = 0

            while not terminal:
                current_state = self.get_state(observation)
                
                if current_state not in self.U:
                    self.U[current_state] = 0.0
                    self.Ns[current_state] = 0
                
                self.Ns[current_state] += 1

                # render method will print you the situation in the terminal
                # self.env.render()
                action = self.receive_observation_and_get_action(observation, terminal)
                observation, reward, terminal, _, _ = self.env.step(action)

                next_state = self.get_state(observation)

                if next_state not in self.U:
                    self.U[next_state] = 0.0
                    self.Ns[next_state] = 0

                alpha = 1 / self.Ns[current_state]

                # Compute value for current state
                self.U[current_state] = self.U.get(current_state) + alpha * (reward + self.discount_factor * self.U.get(next_state) - self.U.get(current_state))


            # self.env.render()

    def receive_observation_and_get_action(self, observation: BlackjackObservation, terminal: bool) -> int:
        return BlackjackAction.HIT.value if observation.player_hand.value() < 17 else BlackjackAction.STAND.value

    def get_hypothesis(self, observation: BlackjackObservation, terminal: bool) -> float:
        """
        Implement this method so that I can test your code. This method is supposed to return your learned U value for
        particular observation.

        :param observation: The observation as in the game. Contains information about what the player sees - player's
        hand and dealer's hand.
        :param terminal: Whether the hands were seen after the end of the game, i.e. whether the state is terminal.
        :return: The learned U-value for the given observation.
        """
        if terminal:
            return 0.0
    
        state = self.get_state(observation)
        return self.U.get(state)
