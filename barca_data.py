import pygame as p
import barca_engine as BarcaEngine
import numpy as np
import barca_AI as AI



WIDTH = HEIGHT = 800
DIMENSION = 10
SQ_SIZE = HEIGHT//DIMENSION
MAX_FPS = 30
#rook is mouse, bishop is lion, queen is elephant
PIECES = {1: 'wR', 2: 'wB', 3: 'wQ',
          -1: 'bR', -2: 'bB', -3: 'bQ'}
IMAGES = {}
WATERING_HOLES = [(3, 3), (3, 6), (6, 3), (6, 6)]
DEPTH = 2

def load_images():
    for number, name in PIECES.items():
        IMAGES[number] = p.transform.scale(p.image.load("images/" + name + '.png'), (SQ_SIZE, SQ_SIZE))


def checkWin(board):
    white_count = 0
    black_count = 0
    for square in WATERING_HOLES:
        if np.sign(board[square]) == 1:
            white_count += 1
        elif np.sign(board[square]) == -1:
            black_count += 1
    if white_count >= 3:
        print('White Wins!')
        return True
    elif black_count >= 3:
        print('Black Wins!')
        return True
    return False



def main():

    p.init()
    height_offset = 0
    screen = p.display.set_mode((WIDTH, HEIGHT+height_offset))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))


    gs = BarcaEngine.GameState()
    load_images()
    while not checkWin(gs.board):
        draw_game_state(screen, gs, False)
        validMoves = gs.getAllValidMoves()
        move = AI.findBestMove(gs, validMoves)
        gs.makeMove(move)
        clock.tick(MAX_FPS)
        p.display.flip()


def draw_game_state(screen, gs, flip_board):
    draw_board(screen)
    draw_text(screen, flip_board)
    draw_pieces(screen, gs.board, flip_board)

    if flip_board:
        screen.blit(p.transform.rotate(screen, 180), (0, 0))

def draw_board(screen):
    colors = [p.Color(235, 235, 208), p.Color(119, 148, 85)]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r+c)%2]
            p.draw.rect(screen, color, p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
    p.draw.rect(screen, p.Color('green'), p.Rect(HEIGHT, 0, 200, WIDTH))
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

def draw_pieces(screen, board, flip_board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r, c]
            if piece != 0:

                if flip_board:
                    screen.blit(p.transform.rotate(IMAGES[piece], 180), p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
                else:
                    screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))



def translate(tup, flipped):
    (x, y) = tup
    #want to take any position on the screen and translate it as if it werent flipped
    if flipped:
        new_x = WIDTH - x
        new_y = HEIGHT - y
    else:
        return x, y
    return new_x, new_y





if __name__ == "__main__":
    main()



