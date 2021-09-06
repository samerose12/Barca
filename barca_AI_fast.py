import numpy as np
from numba import njit



DIMENSION = 10
#rook is mouse, bishop is lion, queen is elephant

WATERING_HOLES = np.array([[3, 3], [3, 6], [6, 3], [6, 6]])

DEPTH = 4

Q_DEPTH = 4

#for hashes, just make a list of the zobrist hash, then np.where will give the index quickly, and we can have other
    #lists for the info that we want

rng = np.random.default_rng(918237456)
zob = np.array([[[rng.integers(low=1, high=np.iinfo(np.int64).max) for _ in range(12)]for _ in range(10)]for _ in range(10)])
del rng
indexing = {1:0, 2:1, 3:2, -1:3, -2:4, -3:5}



@njit
def scoreBoard(board, piece_locations, alpha, beta):
    #lazy_val = 5
    '''
    scaring pieces in good positions is worth more

    scaring pieces that are worth more is worth more
    '''
    #board = bboard.copy()
    win = checkWin(board)
    if win:
        return win*np.inf

    M_position_mask = np.ones((10, 10))
    M_position_mask[1:-1, 1:-1] += 1
    M_position_mask[3:-3, 3:-3] += 2
    M_position_mask[4:-4, 4:-4] -= 1

    L_position_mask = np.ones((10, 10))
    L_position_mask[1:-1, 1:-1] += 1
    #L_position_mask[3:-3, 3:-3] += 1
    L_position_mask[4:-4, 4:-4] += 2

    #get positional score
    score = 0
    w_sight = 0
    b_sight = 0
    for square in piece_locations:

        r, c = int(square[0]), int(square[1])
        piece = abs(board[r, c])
        turn = np.sign(board[r, c])

        if piece == 1:
            #mouse
            score += M_position_mask[r, c] * turn
        elif piece == 3:
            #elephant
            score += L_position_mask[r, c] * turn
            score += M_position_mask[r, c] * turn
        if piece == 2:
            #lion
            score += L_position_mask[r, c] * turn

        for hole in WATERING_HOLES:
            #if you are in a hole
            if int(hole[0]) == r and int(hole[1]) == c:
                score += 5 * turn

            #if you are in sight of a hole
            if piece == 1 or piece == 3:
                # if piece on same rank as a hole
                if hole[0] == r:
                    score += turn
                # if piece on same file
                if hole[1] == c:
                    score += turn

            # if not a mouse
            if piece == 2 or piece == 3:
                if hole[1] - hole[0] == c - r:
                    score += turn

                if hole[1] + hole[0] == c + r:
                    score += turn

    # if score < alpha - lazy_val or score > beta + lazy_val:
    #     return score

    for square in piece_locations:
        r, c = int(square[0]), int(square[1])
        turn = np.sign(board[r, c])
        moves = getMovesForPiece(board, piece_locations, r, c)

        if turn == 1:
            w_sight += len(moves)
        else:
            b_sight += len(moves)


        for m in moves:
            end = int(m[1][0]), int(m[1][1])
            #if this piece sees an open watering hole
            for hole in WATERING_HOLES:
                if int(hole[0]) == end[0] and int(hole[1]) == end[1]:
                    score += 3 * turn
            #if this piece sees an enemy it can attack
            if checkNeighborsCanScare(board, end[0], end[1]):
                score += 2 * turn

    score += (w_sight)/10 - (b_sight)/10

    return score

@njit
def checkWin(board):

    white_count = 0
    black_count = 0
    for square in WATERING_HOLES:
        b = board[int(square[0]), int(square[1])]
        if np.sign(b) == 1:
            white_count += 1
        elif np.sign(b) == -1:
            black_count += 1
    if white_count >= 3:
        return 1
    elif black_count >= 3:
        return -1
    return 0


@njit
def findBestMove(board, piece_locations, turn):

    validMoves = getAllValidMoves(board, piece_locations, turn)

    validMoves = orderMoves(board, validMoves)
    score, nextMove = findMoveNegaMax(board, piece_locations, validMoves, DEPTH, -np.inf, np.inf, turn, validMoves[0])

    #pd.DataFrame(hashes, index=[0]).to_pickle('ZobristHash')

    return nextMove


# @njit
# def quiescence(board, piece_locations, depth, alpha, beta, turn):
#     if depth == 0:
#         return scoreBoard(board, piece_locations, alpha, beta)
#
#     max_score = -1000
#     validMoves = np.concatenate((getScares(board, piece_locations, turn), getCaptures(board, piece_locations, turn)))
#
#     for move in validMoves:
#         board, piece_locations = makeMove(board, move, piece_locations)
#         turn *= -1
#         score = -quiescence(board, piece_locations, depth - 1, -beta, -alpha, turn)
#         board, piece_locations = undoMove(board, move, piece_locations)
#         turn *= -1
#
#         if score > max_score:
#             max_score = score
#
#         if max_score > alpha:
#             alpha = max_score
#         if alpha >= beta:
#             break
#
#     return max_score

@njit
def getCaptures(board, piece_locations, turn):
    captures = np.zeros((1, 2, 2))
    for square in piece_locations:
        r, c = int(square[0]), int(square[1])
        if np.sign(board[r, c]) == turn:
            moves = getMovesForPiece(board, piece_locations, r, c)
            for m in moves:
                end = int(m[1][0]), int(m[1][1])
                # if this piece sees an enemy it can attack
                board, piece_locations = makeMove(board, m, piece_locations)
                for hole in WATERING_HOLES:
                    if int(hole[0]) == end[0] and int(hole[1]) == end[1]:
                        captures = np.append(captures, m.copy().reshape(-1, 2, 2), axis=0)
                board, piece_locations = undoMove(board, m, piece_locations)
    return captures[1:]

@njit
def getScares(board, piece_locations, turn):
    scares = np.zeros((1, 2, 2))
    for square in piece_locations:
        r, c = int(square[0]), int(square[1])
        if np.sign(board[r, c]) == turn:
            moves = getMovesForPiece(board, piece_locations, r, c)
            for m in moves:
                end = int(m[1][0]), int(m[1][1])
                # if this piece sees an enemy it can attack
                board, piece_locations = makeMove(board, m, piece_locations)
                if checkNeighborsCanScare(board, end[0], end[1]):
                    scares = np.append(scares, m.copy().reshape(-1, 2, 2), axis=0)
                board, piece_locations = undoMove(board, m, piece_locations)

    return scares[1:]

@njit
def orderMoves(board, moves):

    wh = []
    sc = []
    rest = []
    index = 0
    for m in moves:
        end = int(m[1][0]), int(m[1][1])
        piece = abs(board[end])
        watering_hole = False

        for hole in WATERING_HOLES:
            if int(hole[0]) == end[0] and int(hole[1]) == end[1]:
                wh.append(index)
                index += 1
                watering_hole = True
                break
            else:
                #check for sight
                r, c = end[0], end[1]

                if piece == 1 or piece == 3:

                    if hole[0] == r:
                        rest.append(index)
                        index += 1
                        watering_hole = True
                        break

                    elif hole[1] == c:
                        rest.append(index)
                        index += 1
                        watering_hole = True
                        break

                elif piece == 2 or piece == 3:
                    if hole[1] - hole[0] == c - r:
                        rest.append(index)
                        index += 1
                        watering_hole = True
                        break

                    elif hole[1] + hole[0] == c + r:
                        rest.append(index)
                        index += 1
                        watering_hole = True
                        break



        if not watering_hole:
            if checkNeighborsCanScare(board, end[0], end[1]):

                sc.append(index)
                index += 1
            else:
                rest.append(index)
                index += 1
    sc.extend(rest)
    wh.extend(sc)
    return moves[np.array(wh)]


@njit
def quiescence(board, piece_locations, depth, alpha, beta, turn):
    if depth == 0:
        return scoreBoard(board, piece_locations, alpha, beta)

    stand_pat = scoreBoard(board, piece_locations, alpha, beta)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    if alpha == stand_pat:
        return alpha

    validMoves = getScares(board, piece_locations, turn)#np.concatenate((getScares(board, piece_locations, turn), getCaptures(board, piece_locations, turn)))
    for move in validMoves:
        board, piece_locations = makeMove(board, move, piece_locations)
        turn *= -1
        score = -quiescence(board, piece_locations, depth-1, -beta, -alpha, turn)
        board, piece_locations = undoMove(board, move, piece_locations)
        turn *= -1

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
        if alpha == score:
            return alpha
    return alpha


@njit
def findMoveNegaMax(board, piece_locations, validMoves, depth, alpha, beta, turn, nextMove):

    if depth == 0:
        #return turn*quiescence(board, piece_locations, Q_DEPTH, alpha, beta, turn), nextMove
        return turn*scoreBoard(board, piece_locations, alpha, beta), nextMove

    #move ordering: we want to evaluate the best moves first to prune faster
    max_score = -1000
    for move in validMoves:
        board, piece_locations = makeMove(board, move, piece_locations)
        if depth == DEPTH:
            win = checkWin(board)
            if win == turn:
                return win*np.inf, move

        turn *= -1
        nextMoves = getAllValidMoves(board, piece_locations, turn)
        score, nextMove = findMoveNegaMax(board, piece_locations, orderMoves(board, nextMoves), depth-1, -beta, -alpha, turn, nextMove)
        score *= -1
        board, piece_locations = undoMove(board, move, piece_locations)
        turn *= -1


        if score > max_score:
            max_score = score
            if depth == DEPTH:
                nextMove = move

        if max_score > alpha:
            alpha = max_score
        if alpha >= beta:
            break

    return max_score, nextMove











@njit
def makeMove(board, move, piece_locations):
    #move as [[], []]
    #piece_locations as array [[x, y], ..., ]
    #board = bboard.copy()
    start = int(move[0][0]), int(move[0][1])
    end = int(move[1][0]), int(move[1][1])
    piece = board[start]
    board[start] = 0
    board[end] = piece

    #find index :(
    pl = piece_locations
    index = 0
    for loc in piece_locations:
        if int(loc[0]) == int(start[0]) and int(loc[1]) == int(start[1]):
            pl[index] = move[1]
        index += 1


    return board, pl

@njit
def undoMove(board, move, piece_locations):
    #print('undoMove')
    #board = bboard.copy()
    start = int(move[0][0]), int(move[0][1])
    end = int(move[1][0]), int(move[1][1])
    piece = board[end]
    board[start] = piece
    board[end] = 0

    # find index :(
    pl = piece_locations
    index = 0
    for loc in piece_locations:
        if int(loc[0]) == int(end[0]) and int(loc[1]) == int(end[1]):
            pl[index] = move[0]
        index += 1

    return board, pl


@njit
def checkNeighborsForScare(board, r, c):

    r = int(r)
    c = int(c)
    piece = board[r, c]
    turn = np.sign(piece)
    if abs(piece) == 3: piece = 0
    if c == 0:
        if r == 0:
            rw, cw = np.where(board[r:r + 2, c:c + 2] == -turn * (abs(piece) + 1))
        else:
            rw, cw = np.where(board[r - 1:r + 2, c:c + 2] == -turn * (abs(piece) + 1))
    else:
        if r == 0:
            rw, cw = np.where(board[r:r + 2, c - 1:c + 2] == -turn * (abs(piece) + 1))
        else:
            rw, cw = np.where(board[r - 1:r + 2, c - 1:c + 2] == -turn * (abs(piece) + 1))

    return len(rw) > 0 or len(cw) > 0


@njit
def checkNeighborsCanScare(board, r, c):
    r = int(r)
    c = int(c)
    piece = board[r, c]
    turn = np.sign(piece)
    if abs(piece) == 1: piece = 4
    if c == 0:
        if r == 0:
            rw, cw = np.where(board[r:r + 2, c:c + 2] == -turn * (abs(piece) - 1))
        else:
            rw, cw = np.where(board[r - 1:r + 2, c:c + 2] == -turn * (abs(piece) - 1))
    else:
        if r == 0:
            rw, cw = np.where(board[r:r + 2, c - 1:c + 2] == -turn * (abs(piece) - 1))
        else:
            rw, cw = np.where(board[r - 1:r + 2, c - 1:c + 2] == -turn * (abs(piece) - 1))

    return len(rw) > 0 or len(cw) > 0


@njit
def getNonValidMoves(board, piece_locations, turn):
    #board = bboard.copy()
    #print('getNonValidMoves')
    attacked = np.zeros((1, 2))
    for square in piece_locations:
        r = int(square[0])
        c = int(square[1])
        if np.sign(board[r, c]) == turn:
            if checkNeighborsForScare(board, r, c):
                attacked = np.append(attacked, np.array([r, c]).reshape(-1, 2), axis=0)
    return attacked[1:]

@njit
def getMovesForPiece(board, piece_locations, r, c):
    #board = bboard.copy()
    #print('getMovesForPiece')
    moves = np.zeros((1, 2, 2))
    r = int(r)
    c = int(c)
    val = board[r, c]
    turn = np.sign(val)
    piece = abs(val)  # gives unsigned piece
    # mouse
    if piece == 1:
        moves = np.append(moves, getRookMoves(board, r, c).reshape(-1, 2, 2), axis=0)
    # lion
    elif piece == 2:
        moves = np.append(moves, getBishopMoves(board, r, c).reshape(-1, 2, 2), axis=0)
    # elephant
    elif piece == 3:
        moves = np.append(moves, getRookMoves(board, r, c).reshape(-1, 2, 2), axis=0)
        moves = np.append(moves, getBishopMoves(board, r, c).reshape(-1, 2, 2), axis=0)

    clean_moves = np.zeros((1, 2, 2))
    for move in moves[1:]:
        board, piece_locations = makeMove(board, move, piece_locations)

        end = int(move[1][0]), int(move[1][1])

        #if len(np.where(np.sign(board[end[0] - 1:end[0] + 2, end[1] - 1:end[1] + 2]) == -turn)[0]) > 0:
        if not checkNeighborsForScare(board, end[0], end[1]):
            clean_moves = np.append(clean_moves, move.copy().reshape(-1, 2, 2), axis=0)
    
        board, piece_locations = undoMove(board, move, piece_locations)


    return clean_moves[1:]

@njit
def getAllValidMoves(board, piece_locations, turn):
    #board = bboard.copy()

    moves = np.zeros((1, 2, 2))

    currently_attacked = getNonValidMoves(board, piece_locations, turn)
    if len(currently_attacked) > 0:
        for square in currently_attacked:
            r = int(square[0])
            c = int(square[1])
            if np.sign(board[r, c]) == turn:
                moves = np.append(moves, getMovesForPiece(board, piece_locations, r, c).reshape(-1, 2, 2), axis=0)


    if len(currently_attacked) == 0 or len(moves) == 1:
        for square in piece_locations:
            r = int(square[0])
            c = int(square[1])
            if np.sign(board[r, c]) == turn:
                moves = np.append(moves, getMovesForPiece(board, piece_locations, r, c).reshape(-1, 2, 2), axis=0)
    return moves[1:]

@njit
def getRookMoves(board, r, c):
    moves = np.zeros((1, 2, 2))
    below = np.where(board[r:, c] == 0)[0]
    above = (r - np.where(board[:r, c] == 0)[0])[::-1]
    left = (c - np.where(board[r, :c] == 0)[0])[::-1]
    right = np.where(board[r, c:] == 0)[0]
    all_arr = [above, below, left, right]

    i = 0
    for a in all_arr:
        w = np.where(np.array([a[j] == j + 1 for j in range(len(a))]))[0]
        for dist in a[w]:
            if i == 0:
                #above
                moves = np.append(moves, np.array([[r, c], [r - dist, c]]).reshape(-1, 2, 2), axis=0)
            elif i == 1:
                #below
                moves = np.append(moves, np.array([[r, c], [r + dist, c]]).reshape(-1, 2, 2), axis=0)
            elif i == 2:
                #left
                moves = np.append(moves, np.array([[r, c], [r, c-dist]]).reshape(-1, 2, 2), axis=0)
            elif i == 3:
                #right
                moves = np.append(moves, np.array([[r, c], [r, c+dist]]).reshape(-1, 2, 2), axis=0)
        i += 1

    return moves[1:]

@njit
def getBishopMoves(board, r, c):
    #board = bboard.copy()

    moves = np.zeros((1, 2, 2))
    downright = np.diag(board[r:, c:])
    upleft = np.diag(np.fliplr(np.flipud(board[:r + 1, :c + 1])))
    upright = np.diag(np.flipud(board[:r + 1, c:]))
    downleft = np.diag(np.fliplr(board[r:, :c + 1]))

    all_arr = [np.where(downright == 0)[0], np.where(upleft == 0)[0], np.where(upright == 0)[0],
               np.where(downleft == 0)[0]]

    i = 0
    for a in all_arr:
        w = np.where(np.array([a[i] == i + 1 for i in range(len(a))]))[0]
        for dist in a[w]:
            if i == 0:
                # down right
                moves = np.append(moves, np.array([[r, c], [r + dist, c + dist]]).reshape(-1, 2, 2), axis=0)
            elif i == 1:
                # upleft
                moves = np.append(moves, np.array([[r, c], [r - dist, c - dist]]).reshape(-1, 2, 2), axis=0)
            elif i == 2:
                # up right
                moves = np.append(moves, np.array([[r, c], [r - dist, c + dist]]).reshape(-1, 2, 2), axis=0)
            elif i == 3:
                # down left
                moves = np.append(moves, np.array([[r, c], [r + dist, c - dist]]).reshape(-1, 2, 2), axis=0)
        i += 1
    return moves[1:]

@njit
def computeHash(piece_locations):
    h = 0
    for piece, loc in piece_locations.items():
        for i, j in loc:
            h ^= zob[i][j][indexing[piece]]
    return h