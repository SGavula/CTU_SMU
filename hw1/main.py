from abstractagent import AbstractAgent
from dealeragent import DealerAgent
from evaluate import *
import gymnasium as gym
from gymnasium import wrappers
from gymnasium.envs.registration import register
from randomagent import RandomAgent
from sarsaagent import SarsaAgent
from tdagent import TDAgent
from plot_table import plot_tables

#import warnings
#warnings.filterwarnings("ignore", message="`np.bool8` is a deprecated alias for `np.bool_`")


def get_env() -> gym.Wrapper:
    """
    Creates the environment. Check the OpenAI Gym documentation.

    :rtype: Environment of the blackjack game that follows the OpenAI Gym API.
    """
    environment = gym.make('smu-blackjack-v2', render_mode="ansi")
    return wrappers.RecordEpisodeStatistics(environment, buffer_length=1000000)


if __name__ == "__main__":
    # Registers the environment so that it can be used
    register(
        id='smu-blackjack-v2',
        entry_point='blackjack:BlackjackEnv'
    )
    # ######################################################
    # IMPORTANT: do not modify the code above this line! ###
    # ######################################################

    # here you can play with the code yourself
    # for example you may want to split the code to two phases - training and testing
    # or you may want to compare two agents
    # feel free to modify the number of games played (highly recommended!)
    # ... or whatever

    env = get_env()
    # number_of_episodes = 100  # TODO do not forget to change the number of episodes
    # number_of_episodes = 1000000  # TODO do not forget to change the number of episodes
    number_of_episodes = 100000  # TODO do not forget to change the number of episodes

    # agent: AbstractAgent = RandomAgent(env, number_of_episodes)
    # agent: AbstractAgent = DealerAgent(env, number_of_episodes)
    agent: AbstractAgent = TDAgent(env, number_of_episodes)
    # agent: AbstractAgent = SarsaAgent(env, number_of_episodes)
    # agent: AbstractAgent = AdvancedAgent(env, number_of_episodes)
    agent.train()

    # in evaluate.py are some ideas that you might want to use to evaluate the agent
    # feel free to modify the code as you want to
    evaluate(list(env.return_queue), number_of_episodes)

    # plot_tables(agent.Q_table)

    # for state, q_values in sorted(agent.Q_table.items()):
    #     best_action = "STAND" if np.argmax(q_values) == 0 else "HIT"
    #     print(f"State {state}: Best Action is {best_action} (Values: {q_values})")