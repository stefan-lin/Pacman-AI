# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    """
      Approach:
        (start with agent 0 - pacman)
        (1) get all possible actions by getLegalActions function
        (2) get all possible successors for each action by generateSuccessor 
            function
        (3) iterate through all possible actions(with its successors) and 
            apply them to minimax function to get result value for each of
            move. Then return the max value we found
    """
    # initial agent index = 0, depth = 0
    return self.__minimax(0, gameState, 0)[0]

  def __minimax(self, agent, game_state, depth):
    """
      FUNCTION __minimax

      In __minimax function, we will increment searching depth only if we iterate
      through all agents and, also, reset the agent index at the same time. The
      __minimax function will be recursively called until we find the best route.
      The terminating conditions for the recursive calls are:
        (1) searching depth equals to self.depth
        (2) win
        (3) lost
        (4) no more legal actions
      In every __minimax calls, the initial value will be initialized to either
      positive infinity or negative infinity corresponding to agent index.
      To iterate all possible actions, I will first zip action and game state
      together and sending the pair to the following recursive call. Right after
      returning from the recursive call, I will store the the return value and
      the action into dictionary (book-keeping for later usage). And update
      the value variable according to minimizer or maximizer. At the end of the
      for loop, return the key-value pair (value, action)

      :param agent: current playing agent (0: pacman else: ghost)
      :param game_state: current game state
      :param depth: depth of search
      :return: (action, recurring_value)
    """
    if agent == game_state.getNumAgents():
      agent = 0
      depth += 1
    # TERMINATING CONDITIONS (FOR RECURSION)
    # REACHED THE END OF DEPTH OR LEAF NODE
    if depth == self.depth or game_state.isWin() or \
      game_state.isLose() or not game_state.getLegalActions(agent):
      return (Directions.STOP, self.evaluationFunction(game_state))

    value   = float("-inf") if agent == 0 else float("inf")
    choices = dict()   # {value: action, value: action, ...}
    for successor in [(action, game_state.generateSuccessor(agent, action))
                  for action in game_state.getLegalActions(agent)]:
      temp = self.__minimax(agent+1, successor[1], depth)
      choices[temp[1]] = successor[0]
      value = max(value, temp[1]) if agent == 0 else min(value, temp[1])
    # END FOR LOOP
    return (choices[value], value)
  # END FUNCTION __minimax

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    AlphaBetaAgent Class
  """
  """
    For AlphaBetaAgent, I had applied the algorithm on Wikipedia.
    (https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
    And (http://oleksiirenov.blogspot.com/2015/03/designing-agents-algorithms-for-pacman.html)
    Had tried to implement like the minimax I did above but it's not successful.
  """
  def getAction(self, gameState):
    # initial agent index = 0, depth = 0, alpha = negative infinity,
    # beta = positive infinity
    return self.__ab_pruning(gameState, 0, 0, float("-inf"), float("inf"))[0]


  def __ab_pruning(self, game_state, agent, depth, alpha, beta):
    """
      FUNCTION __ab_pruning

      The AlphaBeta Pruning is an extended algorithm from MiniMax algorithm.
      The only different part is in the if-else statement (in for loop) which
      will be updating value variable, alpha, and beta. After done updating,
      pass alpha and beta to the recursive calls.

      :param game_state: current game state
      :param agent: current agent index
      :param depth: current searching depth
      :param alpha: the best already explored option along path to the root for
                    maximizer
      :param beta: the best already explored option along path to the root for
                   minimizer
      :return: (action, recurring value) pair
    """
    if agent == game_state.getNumAgents():
      agent = 0
      depth += 1
    if depth == self.depth or game_state.isWin() or \
      game_state.isLose() or not game_state.getLegalActions(agent):
      return self.evaluationFunction(game_state)

    value = (Directions.STOP, float("-inf") if agent == 0 else float("inf"))
    for action in game_state.getLegalActions(agent):
      temp = self.__ab_pruning(game_state.generateSuccessor(agent, action),
                               agent+1, depth, alpha, beta)
      if agent == 0:
        ret_value = max(value[1], temp if type(temp) is not tuple else temp[1])
        value = (action, ret_value) if ret_value is not value[1] else value
        if value[1] > beta:            # SKIP THE SUB-TREE
          return value
        alpha = max(alpha, value[1])   # UPDATING alpha VALUE
      else:
        ret_value = min(value[1], temp if type(temp) is not tuple else temp[1])
        value = (action, ret_value) if ret_value is not value[1] else value
        if value[1] < alpha:           # SKIP THE SUB-TREE
          return value
        beta = min(beta, value[1])     # UPDATING beta VALUE
      # END IF-ELSE
    # END FOR LOOP
    return value
  # END FUNCTION __ab_pruning
# END CLASS
