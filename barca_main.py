import pygame as p
import barca_engine as BarcaEngine
import numpy as np
import barca_AI_fast as AI
#import barca_AI as AI
#import concurrent.futures


WIDTH = HEIGHT = 800
DIMENSION = 10
SQ_SIZE = HEIGHT//DIMENSION
MAX_FPS = 60
#rook is mouse, bishop is lion, queen is elephant
PIECES = {1: 'wM', 2: 'wL', 3: 'wE',
          -1: 'bM', -2: 'bL', -3: 'bE'}
IMAGES = {}
WATERING_HOLES = [(3, 3), (3, 6), (6, 3), (6, 6)]


def load_images():
    for number, name in PIECES.items():
        IMAGES[number] = p.transform.scale(p.image.load("images/" + name + '.png'), (SQ_SIZE, SQ_SIZE))


def main():
    p.init()
    height_offset = 0
    screen = p.display.set_mode((WIDTH, HEIGHT+height_offset))
    clock = p.time.Clock()
    screen.fill(p.Color("grey"))


    gs = BarcaEngine.GameState()
    validMoves = gs.getAllValidMoves()
    moveCounter = 0

    load_images()
    running = True

    sq_selected = ()
    player_clicks = [] #two tuples
    mouse_down = False
    moving_piece = ()

    computer_should_play = True
    automove = False

    moves_to_highlight = []

    flip_board = False

    while running:
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False

            if automove and computer_should_play:
                automove = False

                r, c = np.where(gs.board != 0)
                piece_locations = np.c_[r, c]
                turn = 1 if gs.white_to_move else -1

                move = AI.findBestMove(gs.board.copy(), piece_locations, turn)
                move = np.array(move, dtype=np.int)

                m = BarcaEngine.Move(move[0], move[1], gs)
                gs.makeMove(m)

                print(m.getNotation())


                # validMoves = gs.getAllValidMoves()
                # move = AI.findBestMove(gs, validMoves)
                # gs.makeMove(move)
                # print(move.getNotation())


                moveCounter += 1
                # WIN CONDITION
                white_count = 0
                black_count = 0

                for square in WATERING_HOLES:
                    if np.sign(gs.board[square]) == 1:
                        white_count += 1
                    elif np.sign(gs.board[square]) == -1:
                        black_count += 1
                if white_count >= 3:
                    print('White Wins!')
                elif black_count >= 3:
                    print('Black Wins!')
                else:
                    validMoves = gs.getAllValidMoves()
                    moveCounter += 1
            # else:
            #     with concurrent.futures.ThreadPoolExecutor() as executor:
            #         r, c = np.where(gs.board != 0)
            #         piece_locations = np.c_[r, c]
            #         turn = 1 if gs.white_to_move else -1
            #         future = executor.submit(AI.findBestMove, gs.board.copy(), piece_locations, turn)
            #         arrow_move = future.result()
            #         print(arrow_move)

            #mouse down
            if e.type == p.MOUSEBUTTONDOWN and mouse_down is False:
                #if e.button == 1: #the left is 1, the right is 3
                location = translate(p.mouse.get_pos(),flip_board) #(x, y)
                if location[0] > WIDTH or location[0] < 0 or location[1] > HEIGHT or location[1] < 0: continue
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE
                piece = gs.board[row, col]
                if piece != 0:
                    mouse_down = True
                    sq_selected = (row, col)
                    player_clicks.append(sq_selected)
                    #print(sq_selected)
                    moving_piece = sq_selected
                    moves_to_highlight = gs.getMovesForPiece(sq_selected[0], sq_selected[1], valid_moves=validMoves, from_valid=True)


            #mouse up
            if mouse_down and e.type == p.MOUSEBUTTONUP:

                location = translate(p.mouse.get_pos(), flip_board)  # (x, y)
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE
                if sq_selected == (row, col):
                    #then want to do nothing
                    mouse_down = False
                    sq_selected = ()
                    player_clicks = []
                    moving_piece = ()
                    moves_to_highlight = []

                elif len(player_clicks) > 0:
                    #just finished a move
                    move = BarcaEngine.Move(player_clicks[0], (row, col), gs)
                    if move in validMoves:
                        automove = True
                        mouse_down = False
                        gs.makeMove(move)
                        print(move.getNotation())
                        sq_selected = ()
                        player_clicks = []
                        moving_piece = ()
                        moves_to_highlight = []
                        # WIN CONDITION
                        white_count = 0
                        black_count = 0

                        for square in WATERING_HOLES:
                            if np.sign(gs.board[square]) == 1:
                                white_count += 1
                            elif np.sign(gs.board[square]) == -1:
                                black_count += 1
                        if white_count >= 3:
                            print('White Wins!')
                        elif black_count >= 3:
                            print('Black Wins!')
                        else:
                            validMoves = gs.getAllValidMoves()
                            moveCounter += 1


                    else:
                        mouse_down = False
                        sq_selected = ()
                        player_clicks = []
                        moving_piece = ()
                        moves_to_highlight = []

            if mouse_down:
                #want to pick up the pieces, glue it to mouse
                location = p.mouse.get_pos()  # (x, y)
                moving_piece = (location[0], location[1])


            #key handlers
            if e.type == p.KEYDOWN:
                if e.key == p.K_LEFT:
                    gs.undoMove()
                    validMoves = gs.getAllValidMoves()



                if e.key == p.K_b:
                    flip_board = not flip_board
                    automove = True


        #in while loop, after event checks
        draw_game_state(screen, gs, player_clicks, flip_board)

        #draw previous move highlight
        if len(gs.moveLog) > 0:
            last_move = gs.moveLog[-1]
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(50)
            s.fill((255, 255, 0))
            if flip_board:
                screen.blit(s, translate(((last_move.endCol+1) * SQ_SIZE, (last_move.endRow+1) * SQ_SIZE), flip_board))
            else:
                screen.blit(s, (last_move.endCol * SQ_SIZE, last_move.endRow * SQ_SIZE))
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(50)
            s.fill((255, 255, 0))
            if flip_board:
                screen.blit(s, translate(((last_move.startCol+1) * SQ_SIZE, (last_move.startRow+1) * SQ_SIZE), flip_board))
            else:
                screen.blit(s, (last_move.startCol * SQ_SIZE, last_move.startRow * SQ_SIZE))

        #draw green squares for where current piece can move
        if moving_piece and mouse_down:
            try:
                for m in moves_to_highlight:
                    s = p.Surface((SQ_SIZE, SQ_SIZE))
                    s.set_alpha(100)
                    s.fill((0, 255, 0))
                    if flip_board:
                        screen.blit(s, translate(((m.endCol + 1)*SQ_SIZE, (m.endRow + 1)*SQ_SIZE), flip_board))
                    else:
                        screen.blit(s, (m.endCol*SQ_SIZE, m.endRow*SQ_SIZE))
                    #p.draw.rect(screen, p.Color(0, 255, 0, a=100), p.Rect(m.endCol*SQ_SIZE, m.endRow*SQ_SIZE, SQ_SIZE, SQ_SIZE))

                screen.blit(IMAGES[gs.board[player_clicks[0]]], p.Rect(moving_piece[0] - SQ_SIZE//2, moving_piece[1] - SQ_SIZE//2, SQ_SIZE, SQ_SIZE))
            except Exception as e:
                print(e)


        #highlight checking pieces in red
        if len(gs.moveLog) > 0:
            attackers = gs.getNonValidMoves()
            if len(attackers) > 0:
                #attackers = [a.getPos() for a in attackers]

                for r, c in attackers:
                    s = p.Surface((SQ_SIZE, SQ_SIZE))
                    s.set_alpha(100)
                    s.fill((255, 0, 0))
                    if flip_board:
                        screen.blit(s, translate(((c+1) * SQ_SIZE, (r+1) * SQ_SIZE), flip_board))
                    else:
                        screen.blit(s, (c * SQ_SIZE, r * SQ_SIZE))





        clock.tick(MAX_FPS)
        p.display.flip()


def draw_game_state(screen, gs, skip_piece, flip_board):
    #for skip piece we want the first entry of player clicks if it exists
    skip_piece = (skip_piece[0][0], skip_piece[0][1]) if skip_piece else ()
    draw_board(screen)
    draw_text(screen, flip_board)
    draw_pieces(screen, gs.board, skip_piece, flip_board)

    # font = p.font.Font(None, 20)
    # last_move = gs.moveLog[-1].getNotation() if len(gs.moveLog) > 0 else ''
    # move_text = font.render(f'Last Move: {last_move}', 1, (10, 10, 10))
    # size = move_text.get_size()
    # textpos = (HEIGHT+size[0], WIDTH)
    # screen.blit(move_text, textpos)
    if flip_board:
        screen.blit(p.transform.rotate(screen, 180), (0, 0))

def draw_board(screen):
    #colors = [p.Color(235, 235, 208), p.Color(119, 148, 85)]
    colors = [p.Color(252, 204, 116), p.Color(138, 120, 93)]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r+c)%2]
            p.draw.rect(screen, color, p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
    #p.draw.rect(screen, p.Color('green'), p.Rect(HEIGHT, 0, 200, WIDTH))
    for r, c in WATERING_HOLES:
        p.draw.circle(screen, p.Color(0, 0, 255), ((c + 0.5) * SQ_SIZE, (r + 0.5) * SQ_SIZE), SQ_SIZE / 2, width=3)

def draw_text(screen, flip_board):
    '''
    Can definitely save time here
    '''
    ranks = 'abcdefghij'
    files = 'T987654321'
    if flip_board:
        ranks = 'abcdefghij'[::-1]
        files = 'T987654321'[::-1]
    font = p.font.Font(None, 20)
    for r in range(DIMENSION):
        for c in range(DIMENSION):

            if c == 0:
                #want to put in top left of square
                string = files[r] if files[r] != 'T' else '10'
                text = font.render(string, 1, (10, 10, 10))
                size = text.get_size()
                if flip_board:
                    textpos = translate((5*size[0]//4, r*SQ_SIZE + 5*size[1]//4), flip_board)
                    text = p.transform.rotate(text, 180)
                else:
                    textpos = (size[0]//4, r*SQ_SIZE + size[1]//4)
                screen.blit(text, textpos)
            if r == DIMENSION - 1:
                #want to put in top left of square
                text = font.render(ranks[c], 1, (10, 10, 10))
                size = text.get_size()
                if flip_board:
                    textpos = translate(((c+1)*SQ_SIZE - size[0]//4, (r+1)*SQ_SIZE), flip_board)
                    text = p.transform.rotate(text, 180)
                else:
                    textpos = ((c+1)*SQ_SIZE - 5*size[0]//4, (r+1)*SQ_SIZE - size[1])
                screen.blit(text, textpos)

def draw_pieces(screen, board, skip_piece, flip_board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r, c]
            if piece != 0:
                if (r, c) != skip_piece:
                    if flip_board:
                        screen.blit(p.transform.rotate(IMAGES[piece], 180), p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
                    else:
                        screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def draw_arrow(screen, start, end, color, flip_board):
    r, c = start
    endRow, endCol = end
    center_of_attacker = np.array(((c + 0.5) * SQ_SIZE, (r + 0.5) * SQ_SIZE))
    center_of_move = np.array(((endCol + 0.5) * SQ_SIZE, (endRow + 0.5) * SQ_SIZE))
    direction = center_of_move - center_of_attacker
    direction /= np.linalg.norm(direction)
    direction *= SQ_SIZE / 6

    left_edge = np.zeros((2,))
    right_edge = np.zeros((2,))
    x, y = direction
    left_edge[0] = y
    left_edge[1] = -x
    right_edge[0] = -y
    right_edge[1] = x

    left_edge += center_of_move - direction
    right_edge += center_of_move - direction

    p.draw.line(screen, color, center_of_attacker, center_of_move - direction * 0.9, width=10)

    p.draw.polygon(screen, color, (center_of_move.astype(int), left_edge.astype(int), right_edge.astype(int)))

def translate(tup, flipped):
    (x, y) = tup
    #want to take any position on the screen and translate it as if it werent flipped
    if flipped:
        new_x = WIDTH - x
        new_y = HEIGHT - y
    else:
        return x, y
    return new_x, new_y


def scoreBoard(gs):
    position_mask = np.ones((10, 10))
    position_mask[1:-1, 1:-1] += 1
    position_mask[3:-3, 3:-3] += 3

    #get positional score
    score = 0
    for piece, locations in gs.piece_locations.items():
        turn = np.sign(piece)
        if abs(piece) == 1: piece = 4
        for loc in locations:
            score += position_mask[loc] * turn
            if loc in WATERING_HOLES: score += 10* turn
            moves = gs.getMovesForPiece(loc[0], loc[1])
            for m in moves:
                pos = m.getPos()
                #if this piece sees a watering hole + 2
                if pos in WATERING_HOLES: score += 2* turn
                #if this piece sees an enemy it can attack
                if gs.board[pos] == -turn*(abs(piece) - 1): score += 1* turn

    return score


if __name__ == "__main__":
    main()



