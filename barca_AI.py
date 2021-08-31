import numpy as np
import random



WIDTH = HEIGHT = 800
DIMENSION = 10
SQ_SIZE = HEIGHT//DIMENSION
MAX_FPS = 60
#rook is mouse, bishop is lion, queen is elephant
PIECES = {1: 'wR', 2: 'wB', 3: 'wQ',
          -1: 'bR', -2: 'bB', -3: 'bQ'}
IMAGES = {}
WATERING_HOLES = [(3, 3), (3, 6), (6, 3), (6, 6)]

DEPTH = 2


def scoreBoard(gs):
    if checkWin(gs.board):
        turn = 1 if gs.white_to_move else -1
        return turn*np.inf

    position_mask = np.ones((10, 10))
    position_mask[1:-1, 1:-1] += 1
    position_mask[3:-3, 3:-3] += 2

    #get positional score
    score = 0
    w_sight = 0
    b_sight = 0
    for piece, locations in gs.piece_locations.items():
        turn = np.sign(piece)
        if abs(piece) == 1: piece = 4
        for loc in locations:
            score += position_mask[loc] * turn
            if loc in WATERING_HOLES: score += 10* turn
            moves = gs.getMovesForPiece(loc[0], loc[1])
            if turn == 1:
                w_sight += len(moves)
            else:
                b_sight += len(moves)
            for m in moves:
                pos = m.getPos()
                #if this piece sees a watering hole + 2
                if pos in WATERING_HOLES: score += 3* turn
                #if this piece sees an enemy it can attack
                if gs.board[pos] == -turn*(abs(piece) - 1): score += 2* turn
    score += (w_sight)/10 - (b_sight)/10
    return score

def checkWin(board):
    white_count = 0
    black_count = 0
    for square in WATERING_HOLES:
        if np.sign(board[square]) == 1:
            white_count += 1
        elif np.sign(board[square]) == -1:
            black_count += 1
    if white_count >= 3:
        #print('White Wins!')
        return True
    elif black_count >= 3:
        #print('Black Wins!')
        return True
    return False


def findBestMove(gs, validMoves):
    random.shuffle(validMoves)
    score, nextMove = findMoveNegaMax(gs, validMoves, DEPTH, -np.inf, np.inf, 1 if gs.white_to_move else -1, validMoves[0])
    return nextMove


def findMoveNegaMax(gs, validMoves, depth, alpha, beta, turn, nextMove):
    if depth == 0:
        return turn * scoreBoard(gs), nextMove

    #move ordering: we want to evaluate the best moves first to prune faster
    max_score = -np.inf
    for move in validMoves:
        gs.makeMove(move)
        nextMoves = gs.getAllValidMoves()
        score, nextMove = findMoveNegaMax(gs, nextMoves, depth-1, -beta, -alpha, -turn, nextMove)
        score *= -1
        if score > max_score:
            max_score = score
            if depth == DEPTH:
                nextMove = move
        gs.undoMove()
        if max_score > alpha:
            alpha = max_score
        if alpha >= beta:
            break
    return max_score, nextMove
