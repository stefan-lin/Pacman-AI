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

    return self.__minimax(0, range(gameState.getNumAgents()), gameState,
                          self.depth)[1]

  def __minimax(self, agent, agent_list, game_state, depth):
    # TERMINATING CONDITIONS (FOR RECURSION)
    # REACHED THE END OF DEPTH OR LEAF NODE
    if depth < 1 or game_state.isWin() or game_state.isLose():
      return (self.evaluationFunction(game_state), None)

    value   = float("-inf") if agent == 0 else float("inf")
    choices = dict()   # {value: action, value: action, ...}
    # successors = [(action, gameState), (action, gameState),...]
    successors = [(action, game_state.generateSuccessor(agent, action))
                  for action in game_state.getLegalActions(agent)]

    for successor in successors:
      if agent == 0:
        temp = self.__minimax(
          agent_list[agent+1],
          agent_list,
          successor[1],
          depth
        )
        choices[temp[0]] = successor[0]
        value = max(value, temp[0])
      else:
        temp = self.__minimax(
          agent_list[0 if agent == agent_list[-1] else agent+1],
          agent_list,
          successor[1],
          depth-1 if agent == agent_list[-1] else depth
        )
        choices[temp[0]] = successor[0]
        value = min(value, temp[0])
    # END FOR LOOP
    return (value, choices[value])
  # END FUNCTION __minimax


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    For AlphaBetaAgent, I had applied the algorithm on Wikipedia.
    (https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
    And (http://oleksiirenov.blogspot.com/2015/03/designing-agents-algorithms-for-pacman.html)
    Had tried to implement like the minimax I did above but it's not successful.
  """
  def getAction(self, gameState):
    return self.__ab_pruning(
      gameState,
      0,
      0,
      float("-inf"),
      float("inf")
    )[0]


  def __ab_pruning(self, game_state, agent, depth, alpha, beta):
    if agent >= game_state.getNumAgents():
      agent = 0
      depth += 1
    if depth == self.depth or game_state.isWin() or game_state.isLost():
      return self.evaluationFunction(game_state)

    if agent == 0:   # MAX 
      value = ("unknown", float("-inf"))
      if not game_state.getLegalActions(agent):
        return self.evaluationFunction(game_state)
      # successor = (action, game_state)
      for successor in [(action, gameState.generateSuccessor(agent, action))
                  for action in gameState.getLegalActions(agent)]:
        # MAYBE COMMENT OUT
        if successor[0] == "Stop":
          continue
        # temp = (action, value)
        temp = self.__ab_pruning(
          successor[1],
          agent+1,
          depth,
          alpha,
          beta
        )
        if type(temp) is tuple:
          temp = temp[1]
        ret_value = max(value[1], temp)
        if ret_value is not value[1]:
          value = (action, ret_value)
        if value[1] > beta:
          return value
        alpha = min(alpha, value[1])
      # END FOR LOOP
      return value
    else:   # MIN
      value = ("unknown", float("inf"))
      if not game_state.getLegalActions(agent):
        return self.evaluationFunction(game_state)
      # successor = (action, game_state)
      for successor in [(action, gameState.generateSuccessor(agent, action))
                  for action in gameState.getLegalActions(agent)]:
        if successor[0] == "Stop":
          continue
        temp = self.__ab_pruning(
          successor[1],
          agent+1,
          depth,
          alpha,
          beta
        )
        if type(temp) is tuple:
          temp = temp[1]
        ret_value = min(value[1], temp)
        if ret_value is not value[1]:
          value = (action, ret_value)
        if value[1] < alpha:
          return value
        beta = min(beta, value[1])
      # END FOR LOOP
      return value
  # END FUNCTION __ab_pruning
# END CLASS

#######################################
class AlphaBetaAgentGOOD(MultiAgentSearchAgent):
  """
    For AlphaBetaAgent, I had applied the algorithm on Wikipedia.
    (https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
    And (http://oleksiirenov.blogspot.com/2015/03/designing-agents-algorithms-for-pacman.html)
    Had tried to implement like the minimax I did above but it's not successful.
  """
  def getAction(self, gameState):
    curDepth = 0
    currentAgentIndex = 0
    alpha = float("-inf")
    beta  = float("inf")
    val = self.value(
      gameState, 
      currentAgentIndex, 
      curDepth, 
      alpha, 
      beta
    )
    #print "Returning %s" % str(val)
    return val[0]
  def value(self, gameState, currentAgentIndex, curDepth, alpha, beta):
    if currentAgentIndex >= gameState.getNumAgents():
      currentAgentIndex = 0
      curDepth += 1
    if curDepth == self.depth or gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)
    if currentAgentIndex == 0:
      return self.maxValue(gameState, currentAgentIndex, curDepth, alpha, beta)
    else:
      return self.minValue(gameState, currentAgentIndex, curDepth, alpha, beta)

  def minValue(self, gameState, currentAgentIndex, curDepth, alpha, beta):
    v = ("unknown", float("inf"))
    if not gameState.getLegalActions(currentAgentIndex):
      return self.evaluationFunction(gameState)
    for action in gameState.getLegalActions(currentAgentIndex):
      if action == "Stop":
        continue
      retVal = self.value(
        gameState.generateSuccessor(currentAgentIndex, action),
        currentAgentIndex + 1,
        curDepth,
        alpha,
        beta
      )
      if type(retVal) is tuple:
        retVal = retVal[1]
      vNew = min(v[1], retVal)
      if vNew is not v[1]:
        v = (action, vNew)
      if v[1] < alpha:
        #print "Pruning with '%s' from min since alpha is %2.2f" % (str(v), alpha)
        return v
      beta = min(beta, v[1])
      #print "Setting beta to %2.2f" % beta
    #print "Returning minValue: '%s' for agent %d" % (str(v), currentAgentIndex)
    return v
  def maxValue(self, gameState, currentAgentIndex, curDepth, alpha, beta):
    v = ("unknown", -1*float("inf"))
    if not gameState.getLegalActions(currentAgentIndex):
      return self.evaluationFunction(gameState)
    for action in gameState.getLegalActions(currentAgentIndex):
      if action == "Stop":
        continue
      retVal = self.value(
        gameState.generateSuccessor(currentAgentIndex, action),
        currentAgentIndex + 1, curDepth, alpha, beta)
      if type(retVal) is tuple:
        retVal = retVal[1]
      vNew = max(v[1], retVal)
      if vNew is not v[1]:
        v = (action, vNew)
      if v[1] > beta:
        #print "Pruning with '%s' from min since beta is %2.2f" % (str(v), beta)
        return v
      alpha = max(alpha, v[1])
      #print "Setting alpha to %2.2f" % alpha
    #print "Returning maxValue: '%s' for agent %d" % (str(v), currentAgentIndex)
    return v

class AlphaBetaAgentGOOD(MultiAgentSearchAgent):
  """
    For AlphaBetaAgent, I had applied the algorithm on Wikipedia.
    (https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
    And (http://oleksiirenov.blogspot.com/2015/03/designing-agents-algorithms-for-pacman.html)
    Had tried to implement like the minimax I did above but it's not successful.
  """
  def getAction(self, gameState):
    curDepth = 0
    currentAgentIndex = 0
    alpha = float("-inf")
    beta  = float("inf")
    val = self.value(
      gameState, 
      currentAgentIndex, 
      curDepth, 
      alpha, 
      beta
    )
    #print "Returning %s" % str(val)
    return val[0]
  def value(self, gameState, currentAgentIndex, curDepth, alpha, beta):
    if currentAgentIndex >= gameState.getNumAgents():
      currentAgentIndex = 0
      curDepth += 1
    if curDepth == self.depth or gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)
    if currentAgentIndex == 0:
      return self.maxValue(gameState, currentAgentIndex, curDepth, alpha, beta)
    else:
      return self.minValue(gameState, currentAgentIndex, curDepth, alpha, beta)

  def minValue(self, gameState, currentAgentIndex, curDepth, alpha, beta):
    v = ("unknown", float("inf"))
    if not gameState.getLegalActions(currentAgentIndex):
      return self.evaluationFunction(gameState)
    for action in gameState.getLegalActions(currentAgentIndex):
      if action == "Stop":
        continue
      retVal = self.value(
        gameState.generateSuccessor(currentAgentIndex, action),
        currentAgentIndex + 1,
        curDepth,
        alpha,
        beta
      )
      if type(retVal) is tuple:
        retVal = retVal[1]
      vNew = min(v[1], retVal)
      if vNew is not v[1]:
        v = (action, vNew)
      if v[1] < alpha:
        #print "Pruning with '%s' from min since alpha is %2.2f" % (str(v), alpha)
        return v
      beta = min(beta, v[1])
      #print "Setting beta to %2.2f" % beta
    #print "Returning minValue: '%s' for agent %d" % (str(v), currentAgentIndex)
    return v
  def maxValue(self, gameState, currentAgentIndex, curDepth, alpha, beta):
    v = ("unknown", -1*float("inf"))
    if not gameState.getLegalActions(currentAgentIndex):
      return self.evaluationFunction(gameState)
    for action in gameState.getLegalActions(currentAgentIndex):
      if action == "Stop":
        continue
      retVal = self.value(
        gameState.generateSuccessor(currentAgentIndex, action),
        currentAgentIndex + 1, curDepth, alpha, beta)
      if type(retVal) is tuple:
        retVal = retVal[1]
      vNew = max(v[1], retVal)
      if vNew is not v[1]:
        v = (action, vNew)
      if v[1] > beta:
        #print "Pruning with '%s' from min since beta is %2.2f" % (str(v), beta)
        return v
      alpha = max(alpha, v[1])
      #print "Setting alpha to %2.2f" % alpha
    #print "Returning maxValue: '%s' for agent %d" % (str(v), currentAgentIndex)
    return v
############################
#####  def getAction(self, gameState):
#####    """
#####      Returns the minimax action using self.depth and self.evaluationFunction
#####    """
#####    "*** YOUR CODE HERE ***"
#####    def maxvalue(gameState, alpha, beta, depth):
#####      if gameState.isWin() or gameState.isLose() or depth == 0:
#####        return self.evaluationFunction(gameState)
#####      v = -(float("inf"))
#####      legalActions = gameState.getLegalActions(0)
#####      for action in legalActions:
#####        nextState = gameState.generateSuccessor(0, action)
#####        v = max(v, minvalue(nextState, alpha, beta, gameState.getNumAgents() - 1, depth))
#####        if v > beta:
#####          return v
#####        alpha = max(alpha, v)
#####      return v
#####
#####    def minvalue(gameState, alpha, beta, agentindex, depth):
#####      numghosts = gameState.getNumAgents() - 1
#####      if gameState.isWin() or gameState.isLose() or depth == 0:
#####        return self.evaluationFunction(gameState)
#####      v = float("inf")
#####      legalActions = gameState.getLegalActions(agentindex)
#####      for action in legalActions:
#####        nextState = gameState.generateSuccessor(agentindex, action)
#####        if agentindex == numghosts:
#####          v = min(v, maxvalue(nextState, alpha, beta, depth - 1))
#####          if v < alpha:
#####            return v
#####          beta = min(beta, v)
#####        else:
#####          v = min(v, minvalue(nextState, alpha, beta, agentindex + 1, depth))
#####          if v < alpha:
#####            return v
#####          beta = min(beta, v)
#####      return v
#####
#####
#####
#####    legalActions = gameState.getLegalActions(0)
#####    bestaction = Directions.STOP
#####    score = -(float("inf"))
#####    alpha = -(float("inf"))
#####    beta = float("inf")
#####    for action in legalActions:
#####      nextState = gameState.generateSuccessor(0, action)
#####      prevscore = score
#####      score = max(score, minvalue(nextState, alpha, beta, 1, self.depth))
#####      if score > prevscore:
#####        bestaction = action
#####      if score > beta:
#####        return bestaction
#####      alpha = max(alpha, score)
#####    return bestaction
###  def getAction(self, gameState):
###    """
###      Returns the minimax action using self.depth and self.evaluationFunction
###    """
###
###    ###return self.__alpha_beta_pruning(0, range(gameState.getNumAgents()),
###    ###                               gameState, self.depth,
###    ###                               float("-inf"),
###    ###                               float("inf"))[1]
###    best_action = Directions.STOP
###    alpha = float("-inf")
###    beta  = float("inf")
###    value = float("-inf")
###    choices = dict()
###    # successors = [(action, gameState), (action, gameState),...]
###    successors = [(action, gameState.generateSuccessor(0, action))
###                  for action in gameState.getLegalActions(0)]
###    for successor in successors:
###      temp = self.__ab_min(
###        1,
###        range(gameState.getNumAgents()),
###        successor[1],
###        self.depth,
###        alpha,
###        beta
###      )
###      choices[temp] = successor[0]
###      value = max(value, temp)
###      if value >= beta:
###        return choices[value]
###      alpha = max(alpha, value)
###    # END FOR LOOP
###    return choices[value]
###  # END CLASS METHOD getAction

  def __ab_max(self, agent, agent_list, game_state, depth, alpha, beta):
    if depth < 1 or game_state.isWin() or game_state.isLose():
      return (self.evaluationFunction(game_state), None)

    value   = float("-inf")
    for action in game_state.getLegalActions(agent):
      value = max(value, self.__ab_min(
        agent_list[agent+1],
        agent_list,
        game_state.generateSuccessor(agent, action),
        depth,
        alpha,
        beta
      ))
      if value >= beta:
        return value
      alpha = max(value, alpha)
    # END FOR LOOP
    return value
  # END CLASS METHOD __ab_max


  def __ab_min(self, agent, agent_list, game_state, depth, alpha, beta):
    if depth < 1 or game_state.isWin() or game_state.isLose():
      return (self.evaluationFunction(game_state), None)

    value  = float("inf")

    for action in game_state.getLegalActions(agent):
      if agent == agent_list[-1]:
        value = min(value, self.__ab_max(
          0,
          agent_list,
          game_state.generateSuccessor(agent, action),
          depth-1,
          alpha,
          beta
        ))
        if value <= alpha:
          return value
        beta = min(value, beta)
      else:
        value = min(value, self.__ab_min(
          agent_list[agent+1],
          agent_list,
          game_state.generateSuccessor(agent, action),
          depth,
          alpha,
          beta
        ))
        if value <= alpha:
          return value
        beta = min(value, beta)
    # END FOR LOOP
    return value
  # END CLASS METHOD __ab_min


  def __alpha_beta_pruning(self, agent, agent_list, game_state, depth, alpha, beta):
    # TESTING
    #print "(alpha, beta) = (", alpha, ", ", beta, ")"

    # TERMINATING CONDITIONS (FOR RECURSION)
    # REACHED THE END OF DEPTH OR LEAF NODE
    if depth < 1 or game_state.isWin() or game_state.isLose():
      return (self.evaluationFunction(game_state), None)

    if agent == 0:   # MAX NODE
      choices = dict()   # {value: action, value: action, ...}
      value = float("-inf")
      for successor in [(action, game_state.generateSuccessor(agent, action))
                        for action in game_state.getLegalActions(agent)]:
        temp = self.__alpha_beta_pruning(agent_list[agent+1],
                                         agent_list, successor[1],
                                         depth, alpha, beta)
        #print "max_value = max(temp[0], value) = (", temp[0], ", ", value, ")"
        choices[temp[0]] = successor[0]
        value = max(temp[0], value)
        if value >= beta:
          return (value, None)
        alpha = max(alpha, value)
      return (value, choices[value])
    else:
      choices = dict()   # {value: action, value: action, ...}
      value = float("inf")
      for successor in [(action, game_state.generateSuccessor(agent, action))
                        for action in game_state.getLegalActions(agent)]:
        temp = self.__alpha_beta_pruning(
          agent_list[0 if agent == agent_list[-1] else agent+1],
          agent_list, successor[1], depth-1 if agent == agent_list[-1] else depth, alpha, beta)
        #print "max_value = min(temp[0], value) = (", temp[0], ", ", value, ")"
        choices[temp[0]] = successor[0]
        value = min(temp[0], value)
        if value <= alpha:
          return (value, None)
        beta = min(beta, value)
      return (value, choices[value])
    # END IF-ELSE
  # END FUNCTION


def minimaxPrune(agent, agentList, state, depth, evalFunc, alpha, beta):
  
  if depth <= 0 or state.isWin() == True or state.isLose() == True:
    return evalFunc(state)
    
  if agent == 0:
    v = float("-inf")
  else:
    v = float("inf")
          
  actions = state.getLegalActions(agent)
  successors = [state.generateSuccessor(agent, action) for action in actions]
  for successor in successors:
    
    if agent == 0:
      
      v = max(v, minimaxPrune(agentList[agent+1], agentList, successor, depth, evalFunc, alpha, beta))
      if v> beta:
        return v
      alpha = max (alpha, v)
    elif agent == agentList[-1]:
      
      v = min(v, minimaxPrune(agentList[0], agentList, successor, depth - 1, evalFunc, alpha, beta))
      if v<alpha:
        return v
      beta = min(beta, v)
    else:
     
      v = min(v, minimaxPrune(agentList[agent+1], agentList, successor, depth, evalFunc, alpha, beta))
      if v<alpha:
        return v
      beta = min(beta, v)
  return v

"""
  def _minimax(self, agent, agentList, game_state, depth, evaluate_func):
    # TERMINATING CONDITIONS (FOR RECURSION)
    # 1: REACHED THE END OF DEPTH
    if depth < 1:
      return evaluate_func(game_state)
    # 2: LEAF NODE
    if game_state.isWin() == True or game_state.isLose() == True:
      return evaluate_func(game_state)

    value = float("-inf") if agent == 0 else float("inf")

    actions = game_state.getLegalActions(agent)
    successors = [game_state.generateSuccessor(agent, action) for action in actions]
    for j in range(len(successors)):
      successor = successors[j]

      if agent == 0:
        value = max(value, self._minimax(agentList[agent+1], agentList, successor, depth, evaluate_func))
      elif agent == agentList[-1]:
        value = min(value, self._minimax(agentList[0], agentList, successor, depth - 1, evaluate_func))
      else:
        value = min(value, self._minimax(agentList[agent+1], agentList, successor, depth, evaluate_func))

    return value
"""
"""
  def __alpha_beta_prune(self, agent, agent_list, game_state, depth, evaluate_func, alpha, beta):
    # TERMINATING CONDITIONS (FOR RECURSION)
    # REACHED THE END OF DEPTH OR LEAF NODE
    if depth < 1 or game_state.isWin() or game_state.isLose():
      return (evaluate_func(game_state), None)

    value   = float("-inf") if agent == 0 else float("inf")
    choices = dict()   # {value: action, value: action, ...}
    # successors = [(action, gameState), (action, gameState),...]
    successors = [(action, game_state.generateSuccessor(agent, action))
                  for action in game_state.getLegalActions(agent)]

    for successor in successors:
      if agent == 0:
        temp = self.__alpha_beta_prune(agent_list[agent+1], agent_list,
                              successor[1], depth, evaluate_func, alpha, beta)
        if temp[0] > beta:
          return (temp[0], successor[0])
        choices[temp[0]] = successor[0]
        value = max(alpha, temp[0])
      ###elif agent == agent_list[-1]:
      ###  temp = self.__alpha_beta_prune(agent_list[0], agent_list, successor[1],
      ###                        depth-1, evaluate_func, alpha, beta)
      ###  if temp[0] < alpha:
      ###    return (temp[0], successor[0])
      ###  choices[temp[0]] = successor[0]
      ###  value = min(beta, temp[0])
      else:
        temp = self.__alpha_beta_prune(
                          agent_list[0 if agent == agent_list[-1] else agent+1],
                          agent_list, successor[1], depth, evaluate_func,
                          alpha, beta)
        if temp[0] < alpha:
          return (temp[0], successor[0])
        choices[temp[0]] = successor[0]
        value = min(beta, temp[0])
    # END FOR LOOP
    return (value, choices[value])
"""
  


































