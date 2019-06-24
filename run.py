# cython: language_level=3
import pyximport; pyximport.install()

import game
from game import Game, build_board_state, MovePiece, PlacePiece, RandomGame, Solver, SolverRecursive
import os
import time

def solve_start(depth, recursive, load):
    print(depth, recursive, load)

    start = time.time()
    board_state = build_board_state('000000000000000000000000000')
    test_game = Game(1, board_state)

    print(test_game)

    if recursive:
        solver = SolverRecursive(test_game, depth)

        if load:
            if os.path.isfile('saved.pkl'):
                solver.load('saved.pkl')
    else:
        solver = Solver(test_game, depth)

    result,move,_ = solver.solve(test_game)

    for (_,m,rot)  in move:
        test_game = test_game.rotate(rot)
        test_game = test_game.apply_move(m)
        print(test_game)

    if recursive and load:
        solver.save('saved.pkl')

    print(result, move)

    print('Total Time: %.3f' % (time.time()-start))

def solve_pos1(depth, load):
    board_state = build_board_state('100100200000000000000020000')
    test_game = Game(1, board_state)

    print(test_game)

    solver = SolverRecursive(test_game, depth)
    if load:
        solver.load('saved.pkl')
    result,move = solver.solve(test_game)
    print(result, move)

def solve_pos2(depth):
    board_state = build_board_state('100200000000000000000000000')
    test_game = Game(1, board_state)

    print(test_game)

    solver = Solver(test_game, depth)
    result,move = solver.solve(test_game)
    print(result, move)

def solve_pos3(depth):
    board_state = build_board_state('100200000000100000000000000')
    test_game = Game(2, board_state)

    print(test_game)

    solver = Solver(test_game, depth)
    result,move = solver.solve(test_game)
    print(result, move)

def solve_one_move():
    board_state = build_board_state('100100000200200000000000000')
    test_game = Game(1, board_state)

    print(test_game)

    solver = Solver(test_game)
    result,move = solver.solve(test_game)
    print(result, move)

def solve_two_move():
    board_state = build_board_state('100200000200000000100000100')
    test_game = Game(2, board_state)

    print(test_game)

    solver = Solver(test_game)
    result,move = solver.solve(test_game)
    print(result, move)


def random_game():
    board_state = build_board_state('000000000000000000000000000')
    test_game = Game(1, board_state)

    print(test_game)
    play = RandomGame(test_game)
    play.play()

def test_hash():
    board_state = build_board_state('000000000000000000000000000')
    test_game = Game(1, board_state)

    print(hash(test_game))

    d = {test_game: '1'}

    print(d)

    board_state = build_board_state('000000000000000000000000000')
    test_game = Game(1, board_state)

    print(hash(test_game))

    print(test_game in d)

if __name__ == '__main__':
    solve_start(8, True, False)
    #solve_start(8, True, True)
    #solve_pos1(5, False)
    #solve_pos2(3)
    #solve_pos3(2)
