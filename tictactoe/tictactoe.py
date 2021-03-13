"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None

size = 3


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # The player function should take a board state as input, and return which player’s turn it is (either X or O).
    # In the initial game state, X gets the first move. Subsequently, the player alternates with each additional move.
    # Any return value is acceptable if a terminal board is provided as input (i.e., the game is already over).
    # not needed if board == initial_state():
    # not needed    return X

    empty_count = 0
    for i in range(size):
        for j in range(size):
            if board[i][j] == EMPTY:
                empty_count += 1

    if empty_count % 2:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # The actions function should return a set of all of the possible actions that can be taken on a given board.
    # Each action should be represented as a tuple (i, j) where i corresponds to the row of the move (0, 1, or 2)
    # and j corresponds to which cell in the row corresponds to the move (also 0, 1, or 2).
    # Possible moves are any cells on the board that do not already have an X or an O in them.
    # Any return value is acceptable if a terminal board is provided as input.
    possible_actions = set()
    for i in range(size):
        for j in range(size):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))

    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # The result function takes a board and an action as input, and should return a new board state,
    # without modifying the original board.
    # If action is not a valid action for the board, your program should raise an exception.
    # The returned board state should be the board that would result from taking the original input board,
    # and letting the player whose turn it is make their move at the cell indicated by the input action.
    # Importantly, the original board should be left unmodified: since Minimax will ultimately require
    # considering many different board states during its computation.
    # This means that simply updating a cell in board itself is not a correct implementation of the result function.
    # You’ll likely want to make a deep copy of the board first before making any changes.
    if action not in actions(board):
        raise Exception("This move is not valid!")

    board_cp = copy.deepcopy(board)
    board_cp[action[0]][action[1]] = player(board)
    return board_cp


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # The winner function should accept a board as input, and return the winner of the board if there is one.
    # If the X player has won the game, your function should return X. If the O player has won the game,
    # your function should return O.
    # One can win the game with three of their moves in a row horizontally, vertically, or diagonally.
    # You may assume that there will be at most one winner (that is, no board will ever have both players
    # with three-in-a-row, since that would be an invalid board state).
    # If there is no winner of the game (either because the game is in progress, or because it ended in a tie),
    # the function should return None
    # Check rows
    for i in range(size):
        if board[i][0] == board[i][1] == board[i][2]:
            if board[i][0] == X:
                return X
            elif board[i][0] == O:
                return O
            else:
                return None
    # Check columns
    for i in range(size):
        if board[0][i] == board[1][i] == board[2][i]:
            if board[0][i] == X:
                return X
            elif board[0][i] == O:
                return O
            else:
                return None
    # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] or board[0][2] == board[1][1] == board[2][0]:
            if board[1][1] == X:
                return X
            elif board[1][1] == O:
                return O
            else:
                return None

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # The terminal function should accept a board as input, and return a boolean value
    # indicating whether the game is over.
    # If the game is over, either because someone has won the game or because all cells have been filled
    # without anyone winning, the function should return True.
    # Otherwise, the function should return False if the game is still in progress.

    if winner(board):
        return True

    empty_count = 0
    for i in range(size):
        for j in range(size):
            if board[i][j] == EMPTY:
                empty_count += 1
    if empty_count == 0:
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    # The utility function should accept a terminal board as input and output the utility of the board.
    # If X has won the game, the utility is 1. If O has won the game, the utility is -1.
    # If the game has ended in a tie, the utility is 0.
    # You may assume utility will only be called on a board if terminal(board) is True.
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # The minimax function should take a board as input, and return the optimal
    # move for the player to move on that board.
    # The move returned should be the optimal action (i, j) that is one of the allowable actions on the board.
    # If multiple moves are equally optimal, any of those moves is acceptable.
    # If the board is a terminal board, the minimax function should return None.
    if terminal(board):
        return None

    alpha = -math.inf
    beta = math.inf
    if player(board) == X:
        return maxvalue(board, alpha, beta)[1]
    else:
        return minvalue(board, alpha, beta)[1]


def maxvalue(board, alpha, beta):
    # max(X) aims to maximize the score, pick action a in actions(s) that produces highest value of
    # min-value(result(board, a))
    if terminal(board):
        return utility(board), None  # Add blank second output

    v = -math.inf  # reference: https://www.geeksforgeeks.org/python-infinity/
    best_action = None
    for action in actions(board):
        # Implement move selection directly in this function v = min(v, maxvalue(result(board, move)))
        v_tmp = minvalue(result(board, action), alpha, beta)[0]
        if v_tmp > v:
            v = v_tmp
            best_action = action
        # Alpha-Beta pruning was taken from:
        # https://www.hackerearth.com/blog/developers/minimax-algorithm-alpha-beta-pruning/
        if v >= beta:
            return v, best_action
        if v > alpha:
            alpha = v
    return v, best_action


def minvalue(board, alpha, beta):
    # min(O) aims to minimize the score, pick action a in actions(s) that produces lowest value of
    # max-value(result(board, a))
    if terminal(board):
        return utility(board), None  # Add blank second output

    v = math.inf  # reference: https://www.geeksforgeeks.org/python-infinity/
    best_action = None
    for action in actions(board):
        # Implement move selection directly in this function v = max(v, minvalue(result(board, move)))
        v_tmp = maxvalue(result(board, action), alpha, beta)[0]
        if v_tmp < v:
            v = v_tmp
            best_action = action
        # Alpha-Beta pruning was taken from:
        # https://www.hackerearth.com/blog/developers/minimax-algorithm-alpha-beta-pruning/
        if v <= alpha:
            return v, best_action
        if v < beta:
            beta = v
    return v, best_action
