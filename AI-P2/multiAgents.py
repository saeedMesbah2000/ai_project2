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

from math import inf

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        print("called Evaluation Function: ")
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        print("after function")
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newNumFood = successorGameState.getNumFood()
        newGhostPos = successorGameState.getGhostPositions()


        # print("this is ghosts new state : {}" .format(newGhostStates))
        # print("this is new state : {}" .format(successorGameState))  
        # print("this is new positions: {}" .format(newPos))
        # print("this is current : {} " .format(currentGameState.getScore()))
        # print("this is successor : {} ".format(successorGameState.getScore()))
        # print("this is number of foods : {}" .format(successorGameState.getNumFood()))
        # print("this is new foods location : {}" .format(newFood.asList()))
        # print("this is new scared times : {}" .format(newScaredTimes))
        # print("this is score : {} " .format(successorGameState.getScore()))
        # print("this is ghost positions : {}" .format([util.manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos]))


        "*** YOUR CODE HERE ***"
        sum = 0
        if newNumFood == 0 or len(newFood.asList())==0 :
            sum = 100
        
        else :
            for scary in newScaredTimes:
                sum = sum + scary

            foodDistance = min([util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()])
            sum += 1.0 / foodDistance

            ghostScore = 0
            closestGhost = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos])
            if closestGhost < 2:
                ghostScore = +inf
            else :
                ghostScore = 1.0 / closestGhost
            

            sum -= ghostScore
        

            sum += successorGameState.getScore() - currentGameState.getScore()

            # print("this is my options : {}" .format(sum))


        return sum
        

        return successorGameState.getScore()

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
    Your minimax agent (question 2)
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"   

        def minimax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
                return self.evaluationFunction(gameState)
            if agent == 0: 
                print("--- this is Pacman agent !!!")
                maxValue = -inf
                actions = gameState.getLegalActions(agent)
                finalAction = None
                for action in actions:
                    value = minimax(1, depth, gameState.generateSuccessor(agent, action))
                    if value > maxValue:
                        maxValue = value
                        finalAction = action
                # return [int(maxValue), finalAction]
                return maxValue

            else:  
                print("this is Ghost agent !!!")
                nextAgent = agent + 1 
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                   depth += 1

                minValue = +inf
                actions = gameState.getLegalActions(agent)
                finalAction = None
                for action in actions:
                    value = minimax(nextAgent, depth, gameState.generateSuccessor(agent, action))
                    if value < minValue:
                        minValue = value
                        finalAction = action
                # return [minValue, finalAction]
                return minValue

        maxValue = -inf
        finalAction = None
        actions = gameState.getLegalActions(0)
        for action in actions:
            utility = minimax(1, 0, gameState.generateSuccessor(0, action))
            if utility > maxValue:
                maxValue = utility
                finalAction = action
        return finalAction




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabetaminimax(agent, depth, alpha, beta, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
                return self.evaluationFunction(gameState)

            if agent == 0:  
                print("--- this is Pacman agent!!!")
                maxValue = -inf
                finalAction = None
                actions = gameState.getLegalActions(agent);
                for action in actions:
                    value = alphabetaminimax(1, depth, alpha, beta, gameState.generateSuccessor(agent, action))
                    if value > maxValue:
                        maxValue = value
                        finalAction = action
                    if maxValue > beta:
                        return maxValue
                    if maxValue > alpha:
                        alpha = maxValue
                # return [maxValue, finalAction]
                return maxValue

            else:  
                print("this is Ghost agent !!!")
                minValue = +inf
                finalAction = None
                next_agent = agent + 1
                if gameState.getNumAgents() == next_agent:
                    next_agent = 0
                if next_agent == 0:
                    depth += 1
                actions = gameState.getLegalActions(agent)
                for action in actions:
                    value = alphabetaminimax(next_agent, depth, alpha, beta, gameState.generateSuccessor(agent, action))
                    if value < minValue:
                        minValue = value
                        finalAction = action
                    if minValue < alpha:
                        return minValue
                    if minValue < beta:
                        beta = minValue
                # return [minValue, finalAction]
                return minValue

        maxValue = -inf
        finalAction = None
        alpha = -inf
        beta = inf
        actions = gameState.getLegalActions(0)
        for action in actions:
            utility = alphabetaminimax(1, 0, alpha, beta, gameState.generateSuccessor(0, action))
            if utility > maxValue:
                maxValue = utility
                finalAction = action
            # if maxValue > beta:
            #     return maxvalue 
            alpha = max(alpha, maxValue)

        return finalAction

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
                return self.evaluationFunction(gameState)
            if agent == 0: 
                print("--- this is Pacman agent !!!")
                maxValue = -inf
                actions = gameState.getLegalActions(agent)
                # print("||||||||||||||||||||||||||")
                # print("this is legalactions in expectimax : {}" .format(gameState.getLegalActions(0)))
                # print("||||||||||||||||||||||||||")
                finalAction = None
                for action in actions:
                    value = expectimax(1, depth, gameState.generateSuccessor(agent, action))
                    if value > maxValue:
                        maxValue = value
                        finalAction = action
                # return [int(maxValue), finalAction]
                return maxValue

            else:  
                print("this is Ghost agent !!!")
                nextAgent = agent + 1  
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                   depth += 1

                actions = gameState.getLegalActions(agent)
                finalAction = None
                sum = 0
                for action in actions:
                    value = expectimax(nextAgent, depth, gameState.generateSuccessor(agent, action))
                    sum = sum + value
                sum = sum / float(len(gameState.getLegalActions(agent)))
                return sum

        # print("--------------------------")
        # print("this is all legalactions : {}" .format(gameState.getLegalActions(0)))
        # print("this is numAgents : {}" .format(gameState.getNumAgents()))
        # print("--------------------------")
        maxValue = -inf
        finalAction = None
        actions = gameState.getLegalActions(0)
        for action in actions:
            utility = expectimax(1, 0, gameState.generateSuccessor(0, action))
            if utility > maxValue:
                maxValue = utility
                finalAction = action
        return finalAction

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()
    foodList = foods.asList()
    numFood = currentGameState.getNumFood()

    
    # print("this is pacman positions : {}" .format(pacmanPosition))
    # print("this is foods : {}" .format(foods))
    # print("this is ghostStates : {}" .format(ghostStates))
    # print("this is scared times : {}" .format(scaredTimers))
    # print("this is ghost positions : {}" .format(ghostPositions))
    # print("this is food list : {} " .format(foodList))
    
    "*** YOUR CODE HERE ***"

    # numCapsule = len(currentGameState.getCapsules())


    sum = 0
    if numFood == 0 or len(foodList)==0 :
        sum = 100
        
    else :
        # number of times ghosts are scared
        for scary in scaredTimers:
            sum = sum + scary

        # distance to closest food
        foodDistance = min([util.manhattanDistance(pacmanPosition, foodPos) for foodPos in foodList])
        
        ghostScore = 0
        # ghostMargin = 0
        # distance to closest ghost
        closestGhost = min([util.manhattanDistance(pacmanPosition, ghostPos) for ghostPos in ghostPositions])
        # if ghost is so close ( distance is less than 2 block ) than top priority is to run away
        if closestGhost < 2:
            ghostScore = +inf
            # ghostMargin += 1
        else :
            ghostScore = 1.0 / closestGhost

        sum -= ghostScore    
        # sum -= ghostMarginx
        # sum -= numCapsule
        sum += 1 / numFood
        sum += 1 / foodDistance
        sum += currentGameState.getScore()
        

        # print("this is my options : {}" .format(sum))
    return sum

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
