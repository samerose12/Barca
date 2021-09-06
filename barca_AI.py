import numpy as np
import random
import pandas as pd



DIMENSION = 10
#rook is mouse, bishop is lion, queen is elephant
PIECES = {1: 'wM', 2: 'wL', 3: 'wE',
          -1: 'bM', -2: 'bL', -3: 'bE'}

WATERING_HOLES = [(3, 3), (3, 6), (6, 3), (6, 6)]

DEPTH = 2

hashes = pd.read_pickle('ZobristHash')

indexing = {1:0, 2:1, 3:2, -1:3, -2:4, -3:5}

def scoreBoard(gs):
    global hashes
    h = computeHash(gs.piece_locations, zob)

    try:
        return hashes[h].values[0]
    except:
        pass

    if checkWin(gs.board):
        turn = 1 if gs.white_to_move else -1
        hashes[h] = turn*1000
        return turn*1000

    M_position_mask = np.ones((10, 10))
    M_position_mask[1:-1, 1:-1] += 1
    M_position_mask[3:-3, 3:-3] += 1
    M_position_mask[4:-4, 4:-4] -= 1

    L_position_mask = np.ones((10, 10))
    L_position_mask[1:-1, 1:-1] += 1
    # L_position_mask[3:-3, 3:-3] += 1
    L_position_mask[4:-4, 4:-4] += 1

    # get positional score
    score = 0
    w_sight = 0
    b_sight = 0
    piece_locations = np.zeros((1, 2))
    for loc in gs.piece_locations.values():
        piece_locations = np.append(piece_locations, np.array([loc[0][0], loc[0][1]]).reshape(-1, 2),
                                    axis=0)
        piece_locations = np.append(piece_locations, np.array([loc[1][0], loc[1][1]]).reshape(-1, 2),
                                    axis=0)
    for square in piece_locations[1:]:

        r, c = int(square[0]), int(square[1])
        piece = abs(gs.board[r, c])
        turn = np.sign(gs.board[r, c])

        if piece == 1:
            # mouse
            score += M_position_mask[r, c] * turn
        elif piece == 3:
            # elephant
            score += L_position_mask[r, c] * turn
            score += M_position_mask[r, c] * turn
        if piece == 2:
            # lion
            score += L_position_mask[r, c] * turn

        for hole in WATERING_HOLES:
            if int(hole[0]) == r and int(hole[1]) == c:
                score += 5 * turn

        moves = gs.getMovesForPiece(r, c)

        if turn == 1:
            w_sight += len(moves)
        else:
            b_sight += len(moves)

        for m in moves:
            end = m.getPos()

            # if this piece sees a watering hole + 2
            for hole in WATERING_HOLES:
                if int(hole[0]) == end[0] and int(hole[1]) == end[1]:
                    score += 2 * turn

            # if this piece sees an enemy it can attack
            if gs.checkNeighborsCanScare(end[0], end[1]):
                score += 1 * turn

    score += (w_sight) / 10 - (b_sight) / 10

    hashes[h] = score
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
    global hashes, zob
    zob = initZob()
    np.random.shuffle(validMoves)
    score, nextMove = findMoveNegaMax(gs, validMoves, DEPTH, -1000, 1000, 1 if gs.white_to_move else -1, validMoves[0])

    pd.DataFrame(hashes, index=[0]).to_pickle('ZobristHash')

    return nextMove


def findMoveNegaMax(gs, validMoves, depth, alpha, beta, turn, nextMove):
    global zob
    if depth == 0:
        # save hash and score
        pos_hash = computeHash(gs.piece_locations, zob)
        try:
            return turn*hashes[pos_hash].values[0], nextMove
        except:
            hashes[pos_hash] = scoreBoard(gs)
            return turn*scoreBoard(gs), nextMove

    #move ordering: we want to evaluate the best moves first to prune faster
    max_score = -1000
    for move in validMoves:
        gs.makeMove(move)
        nextMoves = gs.getAllValidMoves()

        score, nextMove = findMoveNegaMax(gs, nextMoves, depth-1, -beta, -alpha, -turn, nextMove)
        score *= -1
        if score > max_score:
            max_score = score
            if depth == DEPTH:
                nextMove = move
                if checkWin(gs.board):
                    gs.undoMove()
                    return max_score, nextMove
        gs.undoMove()
        if max_score > alpha:
            alpha = max_score
        if alpha >= beta:
            break
    return max_score, nextMove


def initZob():
    rng = np.random.default_rng(918237456)
    zob = np.array([[[rng.integers(low=1, high=np.iinfo(np.int64).max) for _ in range(len(PIECES))]for _ in range(DIMENSION)]for _ in range(DIMENSION)])
    return zob


def computeHash(piece_locations, zob):
    h = 0
    for piece, loc in piece_locations.items():
        for i, j in loc:
            h ^= zob[i][j][indexing[piece]]
    return h