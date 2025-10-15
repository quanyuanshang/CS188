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
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        scores=successorGameState.getScore()
        foodList = newFood.asList()
        capsules = currentGameState.getCapsules()
        for food in foodList:
            dist=manhattanDistance(newPos,food)
            if dist>0:
                scores+=1/(dist)
        for capsule in capsules:
            dist=manhattanDistance(newPos,capsule)
            if dist>0:
                scores+=2/(dist)
        for ghost in newGhostStates:
            dist=manhattanDistance(newPos,ghost.getPosition())
            if ghost.scaredTimer>0:
                if dist>0:
                    scores+=2/(dist)
            else:
                if 0<dist<3:
                    scores-=200
                elif dist>=3:
                    scores-=2/(dist)
        "*** YOUR CODE HERE ***"

        return scores

def scoreEvaluationFunction(currentGameState: GameState):
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
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
                # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = float('-inf')
        bestaction=0
        for action in legalMoves:
            nextState=gameState.generateSuccessor(0,action)
            if scores<self.minmax(nextState,0,1):
                scores=self.minmax(nextState,0,1)
                bestaction=action
        
        "Add more of your code here if you want to"

        return bestaction
    def minmax(self,gameState,depth,agentIndex):
        if gameState.isWin() or gameState.isLose() or depth==self.depth:
            return self.evaluationFunction(gameState)
        bestscore=float('-inf') if agentIndex==0 else float('inf')
        nextagent=(agentIndex+1)%gameState.getNumAgents()
        nextdepth=depth+1 if nextagent==0 else depth
        legalMoves=gameState.getLegalActions(agentIndex)
        if not legalMoves:
            return self.evaluationFunction(gameState)
        if agentIndex==0:
            for action in legalMoves:
                nextState=gameState.generateSuccessor(agentIndex,action)
                bestscore=max(bestscore,self.minmax(nextState,nextdepth,nextagent))
            
        else:
            for action in legalMoves:
                nextState=gameState.generateSuccessor(agentIndex,action)
                bestscore=min(bestscore,self.minmax(nextState,nextdepth,nextagent))
        return bestscore                
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = float('-inf')
        bestaction=None
        alpha=float('-inf')
        beta=float('inf')
        for action in legalMoves:
            nextState=gameState.generateSuccessor(0,action)
            if scores<self.AlphaBeta(nextState,0,1,alpha,beta):
                scores=self.AlphaBeta(nextState,0,1,alpha,beta)
                bestaction=action
            alpha=max(alpha,scores)
        
        "Add more of your code here if you want to"

        return bestaction
    def AlphaBeta(self,gameState,depth,agentIndex,alpha,beta):
        if gameState.isWin() or gameState.isLose() or depth==self.depth:
            return self.evaluationFunction(gameState)
        bestscore=float('-inf') if agentIndex==0 else float('inf')
        nextagent=(agentIndex+1)%gameState.getNumAgents()
        nextdepth=depth+1 if nextagent==0 else depth
        legalMoves=gameState.getLegalActions(agentIndex)
        if not legalMoves:
            return self.evaluationFunction(gameState)
        if agentIndex==0:
            for action in legalMoves:
                nextState=gameState.generateSuccessor(agentIndex,action)
                if bestscore<self.AlphaBeta(nextState,nextdepth,nextagent,alpha,beta):
                    bestscore=self.AlphaBeta(nextState,nextdepth,nextagent,alpha,beta)
                    if bestscore>beta:
                        return bestscore
                    alpha=max(alpha,bestscore)
            
        else:
            for action in legalMoves:
                nextState=gameState.generateSuccessor(agentIndex,action)
                if bestscore>self.AlphaBeta(nextState,nextdepth,nextagent,alpha,beta):
                    bestscore=self.AlphaBeta(nextState,nextdepth,nextagent,alpha,beta)
                    if bestscore<alpha:
                        return bestscore
                    beta=min(beta,bestscore)
        return bestscore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        legalMoves = gameState.getLegalActions()
        scores = float('-inf')
        bestaction=None
 
        for action in legalMoves:
            nextState=gameState.generateSuccessor(0,action)
            if scores<self.expmax(nextState,0,1):
                scores=self.expmax(nextState,0,1)
                bestaction=action
            
        
        "Add more of your code here if you want to"

        return bestaction
    def expmax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        numAgents = gameState.getNumAgents()
        nextagent = (agentIndex + 1) % numAgents
        nextdepth = depth + 1 if nextagent == 0 else depth
        legalMoves = gameState.getLegalActions(agentIndex)
        if not legalMoves:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            # Pacman: maximize
            bestscore = float('-inf')
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex, action)
                score = self.expmax(nextState, nextdepth, nextagent)
                bestscore = max(bestscore, score)
            return bestscore
        else:
            # Ghost: expectation
            total = 0
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agentIndex, action)
                score = self.expmax(nextState, nextdepth, nextagent)
                total += score
            return total / len(legalMoves)
            
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    scores=successorGameState.getScore()
    foodList = newFood.asList()
    capsules = currentGameState.getCapsules()
    for food in foodList:
        dist=manhattanDistance(newPos,food)
        if dist>0:
            scores+=1/(dist)
    for capsule in capsules:
        dist=manhattanDistance(newPos,capsule)
        if dist>0:
            scores+=2/(dist)
    for ghost in newGhostStates:
        dist=manhattanDistance(newPos,ghost.getPosition())
        if ghost.scaredTimer>0:
            if dist>0:
                scores+=2/(dist)
        else:
            if 0<dist<3:
                scores-=200
            elif dist>=3:
                scores-=2/(dist)
    return scores

# Abbreviation
better = betterEvaluationFunction
