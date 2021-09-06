import numpy as np
from numba import jit

BOARD_SIZE = 10

'''
Empty = 0
Mouse = 1
Lion = 2
Elephant = 3

White is positive, Black is Negative
'''

class GameState():
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype='int8')

        #Board setup at beginning, bottom will be white

        #lets try to constantly keep a piece location dict since we cannot capture pieces
        self.piece_locations = {-1: [(1, 4), (1, 5)], -2: [(1, 3), (1, 6)], -3: [(0, 4), (0, 5)],
                          1: [(8, 4), (8, 5)], 2: [(8, 3), (8, 6)], 3: [(9, 4), (9, 5)]}

        # mice
        self.board[[1, -2], [[4], [5]]] = 1
        # lion
        self.board[[1, -2], [[3], [6]]] = 2
        # elephants
        self.board[[0, -1], [[4], [5]]] = 3


        #set black color
        self.board[:3, :] *= -1


        self.white_to_move = True
        self.moveLog = []

    def makeMove(self, move):
        self.board[move.startRow, move.startCol] = 0
        self.board[move.endRow, move.endCol] = move.piece

        self.moveLog.append(move)
        self.piece_locations[move.piece][move.piece_index] = (move.endRow, move.endCol)

        #change turns
        self.white_to_move = not self.white_to_move


    def undoMove(self):
        if len(self.moveLog) > 0:
            move = self.moveLog.pop()
            #sf_move = self.st_moveLog.pop()
            self.board[move.startRow, move.startCol] = move.piece
            self.board[move.endRow, move.endCol] = 0

            self.piece_locations[move.piece][move.piece_index] = (move.startRow, move.startCol)
            # change turns back
            self.white_to_move = not self.white_to_move


    def checkNeighborsForScare(self, r, c):
        piece = self.board[r, c]
        turn = np.sign(piece)
        if abs(piece) == 3: piece = 0
        if c == 0:
            if r == 0:
                rw, cw = np.where(self.board[r:r + 2, c:c + 2] == -turn * (abs(piece) + 1))
            else:
                rw, cw = np.where(self.board[r - 1:r + 2, c:c + 2] == -turn * (abs(piece) + 1))
        else:
            if r == 0:
                rw, cw = np.where(self.board[r:r + 2, c - 1:c + 2] == -turn * (abs(piece) + 1))
            else:
                rw, cw = np.where(self.board[r - 1:r + 2, c - 1:c + 2] == -turn * (abs(piece) + 1))

        return len(rw) > 0 or len(cw) > 0


    def checkNeighborsCanScare(self, r, c):

        piece = self.board[r, c]
        turn = np.sign(piece)
        if abs(piece) == 1: piece = turn*4
        if c == 0:
            if r == 0:
                rw, cw = np.where(self.board[r:r + 2, c:c + 2] == -turn * (abs(piece) - 1))
            else:
                rw, cw = np.where(self.board[r - 1:r + 2, c:c + 2] == -turn * (abs(piece) - 1))
        else:
            if r == 0:
                rw, cw = np.where(self.board[r:r + 2, c - 1:c + 2] == -turn * (abs(piece) - 1))
            else:
                rw, cw = np.where(self.board[r - 1:r + 2, c - 1:c + 2] == -turn * (abs(piece) - 1))

        return len(rw) > 0 or len(cw) > 0



    def getNonValidMoves(self):
        #turn = 1 if self.white_to_move else -1
        attackers = []
        for piece, sqaures in self.piece_locations.items():
            for r, c in sqaures:
                if self.checkNeighborsForScare(r, c):
                    attackers.append((r, c))
        return attackers


    def getMovesForPiece(self, r, c, valid_moves=[], from_valid = False, getScore=False, score_func=None):

        if from_valid:
            piece_moves = []
            for move in valid_moves:
                if (move.startRow, move.startCol) == (r, c):
                    piece_moves.append(move)
            return piece_moves
        else:
            moves = []
            clean_moves = []
            val = self.board[r, c]

            if getScore:
                boards = []
                scores = []

            turn = np.sign(val)

            piece = abs(val)  # gives unsigned piece

            # mouse
            if piece == 1:
                self.getRookMoves(r, c, turn, moves)
            # lion
            elif piece == 2:
                self.getBishopMoves(r, c, turn, moves)
            # elephant
            elif piece == 3:
                self.getRookMoves(r, c, turn, moves)
                self.getBishopMoves(r, c, turn, moves)

            #dont let king move into check
            temp_board = self.board.copy()

            for move in moves:

                self.board[move.getPos()] = move.piece
                self.board[move.startRow, move.startCol] = 0
                self.piece_locations[move.piece][move.piece_index] = (move.endRow, move.endCol)

                attackers = []
                if not self.checkNeighborsForScare(move.endRow, move.endCol):
                    clean_moves.append(move)
                    if getScore:
                        boards.append(self.board.copy())
                        scores.append(score_func(self))

                self.piece_locations[move.piece][move.piece_index] = (move.startRow, move.startCol)
                self.board = temp_board.copy()

            if getScore:
                return boards, scores
            else:
                return clean_moves


    def getAllValidMoves(self):
        #get all moves
        moves = []
        currently_attacked = self.getNonValidMoves()
        turn = 1 if self.white_to_move else -1
        if len(currently_attacked) > 0:
            for r, c in currently_attacked:
                if np.sign(self.board[r, c]) == turn:
                    moves.extend(self.getMovesForPiece(r, c))


        if len(currently_attacked) == 0 or len(moves) == 0:
            for piece, locations in self.piece_locations.items():
                if np.sign(piece) == turn:
                    for r, c in locations:
                        moves.extend(self.getMovesForPiece(r, c))
        return moves


    def getRookMoves(self, r, c, turn, moves):
        below = np.where(self.board[r:, c] == 0)[0]
        above = (r - np.where(self.board[:r, c] == 0)[0])[::-1]
        left = (c - np.where(self.board[r, :c] == 0)[0])[::-1]
        right = np.where(self.board[r, c:] == 0)[0]
        all_arr = [above, below, left, right]


        for i, a in enumerate(all_arr):
            w = np.where([a[i] == i + 1 for i in range(len(a))])[0]
            for dist in a[w]:
                if i == 0:
                    #above
                    moves.append(Move((r, c), (r - dist, c), self))
                elif i == 1:
                    #below
                    moves.append(Move((r, c), (r + dist, c), self))
                elif i == 2:
                    #left
                    moves.append(Move((r, c), (r, c-dist), self))
                elif i == 3:
                    #right
                    moves.append(Move((r, c), (r, c+dist), self))


    def getBishopMoves(self, r, c, turn, moves):
        downright = self.board[r:, c:].diagonal()
        upleft = np.fliplr(np.flipud(self.board[:r + 1, :c + 1])).diagonal()
        upright = np.flipud(self.board[:r + 1, c:]).diagonal()
        downleft = np.fliplr(self.board[r:, :c + 1]).diagonal()

        all_arr = [np.where(downright == 0)[0], np.where(upleft == 0)[0], np.where(upright == 0)[0],
                   np.where(downleft == 0)[0]]
        for i, a in enumerate(all_arr):
            w = np.where([a[i] == i + 1 for i in range(len(a))])[0]
            for dist in a[w]:
                if i == 0:
                    # down right
                    moves.append(Move((r, c), (r + dist, c + dist), self))
                elif i == 1:
                    # upleft
                    moves.append(Move((r, c), (r - dist, c - dist), self))
                elif i == 2:
                    # up right
                    moves.append(Move((r, c), (r - dist, c + dist), self))
                elif i == 3:
                    # down left
                    moves.append(Move((r, c), (r + dist, c - dist), self))


class Move():
    #attributes
    ranksToRows = {f'{i}': 10 - i for i in range(1, 11)}
    rowsToRanks = {v: k for k, v in ranksToRows.items()}

    filesToCols = {'abcdefghij'[i]: i for i in range(10)}
    colsToFiles = {v: k for k, v in filesToCols.items()}

    pieces = {0: '', 1: 'M', 2: 'L', 3: 'E'}

    def __init__(self, start, end, gs):
        self.startRow = start[0]
        self.startCol = start[1]
        self.endRow = end[0]
        self.endCol = end[1]
        self.piece = gs.board[self.startRow, self.startCol]
        self.moveID = self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol
        self.piece_index = gs.piece_locations[self.piece].index((self.startRow, self.startCol))

    def __eq__(self, other):
        if isinstance(other, Move):
            return self.moveID == other.moveID

    def getNotation(self):

        return self.pieces[abs(self.piece)] + self.getRankFile(self.endRow, self.endCol)

    def boardToNotation(self, pos):
        r, c = pos
        return self.colsToFiles[c] + self.rowsToRanks[r]

    def getRankFile(self, r, c):
        return self.colsToFiles[c] + self.rowsToRanks[r]

    def getPos(self):
        return (self.endRow, self.endCol)
