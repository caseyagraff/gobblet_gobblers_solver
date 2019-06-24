import numpy as np
import time
import pickle
from collections import deque

# 0 -> Draw, 1 -> Player 1 Victory, 2 -> Player 2 Victory, -1 -> None
DRAW = 0
P1_VICTORY = 1
P2_VICTORY = 2
UNDECIDED  = -1

def str_to_byte(cell_str):
    return int(cell_str, 3)

def byte_to_str(cell):
    DIGITS = '012'

    cell_str = ''

    if cell == 0:
        return '000'

    while len(cell_str) < 3:
        cell_str = DIGITS[cell%3] + cell_str
        cell = cell//3

    return cell_str

def get_active_piece(cell):
    # -1 -> No piece, 0 -> Large, 1 -> Medium, 2 -> Small
    # 0 -> No player, 1 -> Player 1, 2 -> Player 2
    piece_type = -1
    player = 0
    for i in range(3):
        if cell[i] != '0':
            piece_type = i
            player = cell[i]
            break

    return int(piece_type), int(player)

def build_board_state(board_state_str):
    board_state = np.zeros(9, dtype='<U3')
    for i in range(9):
        board_state[i] = board_state_str[i*3:(i+1)*3]

    return board_state

def build_board_str(board_state):
    board_state_str = ''
    for i in range(9):
        #board_state_str += byte_to_str(board_state[i])
        board_state_str += board_state[i]

    return board_state_str

DRAW_MOVE_NUM = 60

def get_game_state(board_state, owns, move_num=None):
    # 0 -> Draw, 1 -> Player 1 Victory, 2 -> Player 2 Victory, -1 -> None

    if move_num is not None and move_num >= DRAW_MOVE_NUM:
        return DRAW

    # Check rows
    """
    rows = np.sum(owns, axis=1)
    if any(rows==3):
        return P1_VICTORY
    elif any(rows==-3):
        return P2_VICTORY
    """
    if all(owns[0,:]==1) or all(owns[1,:]==1) or all(owns[2,:]==1):
        return P1_VICTORY
    elif all(owns[0,:]==-1) or all(owns[1,:]==-1) or all(owns[2,:]==-1):
        return P2_VICTORY

    # Check cols
    if all(owns[:,0]==1) or all(owns[:,1]==1) or all(owns[:,2]==1):
        return P1_VICTORY
    elif all(owns[:,0]==-1) or all(owns[:,1]==-1) or all(owns[:,2]==-1):
        return P2_VICTORY

    # Check diag
    diag1 = owns[0,0] + owns[1,1] + owns[2,2]
    diag2 = owns[0,2] + owns[1,1] + owns[2,0]

    if diag1==3 or diag2==3:
        return P1_VICTORY
    elif diag1==-3 or diag2==-3:
        return P2_VICTORY

    return UNDECIDED

ROT_IND = np.array([2,5,8,1,4,7,0,3,6])

def get_player_hands(board_state):
    hands = {1: np.array([2,2,2]), 2: np.array([2,2,2])}

    for i in range(9):
        #cell = byte_to_str(board_state[i])
        cell = board_state[i]
        for j in range(3):
            piece = cell[j]
            if piece == '1':
                hands[1][j] -= 1 
            elif piece == '2':
                hands[2][j] -= 1 

    return hands

def get_valid_placements(board_state, hand, player):
    moves = []

    for i in range(9):
        #cell = byte_to_str(board_state[i])
        cell = board_state[i]
        for j in range(3):
            if cell[j] != '0':
                break
            elif hand[j] > 0:
                moves.append(PlacePiece(j, i))

    return moves

def get_valid_movements(board_state, player):
    moves = []

    active_pieces = []
    for i in range(9):
        cell = board_state[i]
        piece, player_ = get_active_piece(cell)

        if player == player_:
            active_pieces.append((piece,i))

    for i in range(9):
        cell = board_state[i]
        piece, player_ = get_active_piece(cell)

        for (piece_,pos) in active_pieces:
            if pos != i and (piece == -1 or piece_ < piece):
                moves.append(MovePiece(pos,i))

    return moves

def build_owns_table(board_state):
    owns = np.zeros((3,3), dtype=np.int8)
    for i in range(9):
        #_, player = get_active_piece(byte_to_str(board_state[i]))
        _, player = get_active_piece(board_state[i])
        owns[i//3,i%3] = -1 if player == 2 else player

    return owns

def rotate_board_state(board_state, rot_num):
    rot_num = rot_num % 4
    rotated_board_state = board_state

    for i in range(rot_num):
        rotated_board_state = rotated_board_state[ROT_IND]
    
    return rotated_board_state

def rotate_board_state_single(board_state):
    rotated_board_state = board_state[ROT_IND]

    return rotated_board_state

class Game:
    def __init__(self, turn, board_state, move_num=0, hands=None, owns=None):
        self.turn = turn 
        self.board_state = board_state
        self.move_num = move_num
        #self.game_state = get_game_state(self.board_state, self.move_num)
        self.game_state__ = None
        if hands is None:
            self.hands = get_player_hands(self.board_state)
        else:
            self.hands = hands
        if owns is None:
            self.owns = build_owns_table(self.board_state)
        else:
            self.owns = owns

        self.hash__ = None

    def game_state(self):
        if self.game_state__ is None:
            self.game_state__ = get_game_state(self.board_state, self.owns, self.move_num)

        return self.game_state__

    def rotate(self, rot_num):
        if rot_num % 4 == 0:
            return self

        rotated_board = rotate_board_state(self.board_state, rot_num)
        return Game(self.turn, rotated_board, self.move_num, self.hands, owns=None)

    def build_hash(self):
        board_state = self.board_state
        turn_str = str(self.turn)

        hash_str = ''.join([turn_str , board_state[0] , board_state[1] , board_state[2] , board_state[3] , board_state[4] , 
            board_state[5] , board_state[6] , board_state[7] , board_state[8]])

        hash_val = int(hash_str,3)
        min_hash_val = hash_val
        min_hash_rot = 0

        hash_str = ''.join([turn_str , board_state[2] , board_state[5] , board_state[8] , board_state[1] , board_state[4] , 
                   board_state[7] , board_state[0] , board_state[3] , board_state[6]])

        hash_val = int(hash_str,3)

        if hash_val < min_hash_val:
            min_hash_val = hash_val
            min_hash_rot = 1

        hash_str = ''.join([turn_str , board_state[8] , board_state[7] , board_state[6] , board_state[5] , board_state[4] , 
            board_state[3] , board_state[2] , board_state[1] , board_state[0]])

        hash_val = int(hash_str,3)

        if hash_val < min_hash_val:
            min_hash_val = hash_val
            min_hash_rot = 2

        hash_str = ''.join([turn_str , board_state[6] , board_state[3] , board_state[0] , board_state[7] , board_state[4] , 
            board_state[1] , board_state[8] , board_state[5] , board_state[2]])

        hash_val = int(hash_str,3)

        if hash_val < min_hash_val:
            min_hash_val = hash_val
            min_hash_rot = 3

        return min_hash_val, min_hash_rot

    def __hash__(self):
        if self.hash__ is None:
            self.hash__,self.hash_rot__ = self.build_hash()

        return self.hash__

    def hash_rot(self):
        return hash(self),self.hash_rot__

    def apply_move(self, move):
        new_board_state, new_hands, new_owns = move.apply(self.board_state, self.turn, self.hands, self.owns)
        new_turn = 1 if self.turn == 2 else 2

        return Game(new_turn, new_board_state, self.move_num+1, new_hands, new_owns)

    def get_valid_moves(self):
        if self.game_state() != UNDECIDED:
            return []

        hand = self.hands[self.turn]

        valid_placements = get_valid_placements(self.board_state, hand, self.turn)
        valid_movements  = get_valid_movements(self.board_state, self.turn)

        return valid_placements + valid_movements

    def __repr__(self):
        board_state_rem = build_board_str(self.board_state)

        board_str = ''
        if self.game_state() == -1:
            board_str += 'Player %d' % self.turn
        elif self.game_state() == 0:
            board_str += '(DRAW)' % self.turn
        else:
            board_str += '(Player %d WINS)' % self.game_state()

        board_str += '\n'


        for i in range(3):
            board_str += '%s|%s|%s\n' % (board_state_rem[:3], board_state_rem[3:6], board_state_rem[6:9])
            board_state_rem = board_state_rem[9:]

        return board_str

def set_piece(cell, piece, player):
    new_cell = ''
    for i in range(3):
        new_cell += cell[i] if i != piece else str(player)

    return new_cell

class MovePiece:
    def __init__(self, pos1, pos2):
        self.pos1 = pos1
        self.pos2 = pos2

    def apply(self, board_state, player_turn, hands, owns):
        #cell1 = byte_to_str(board_state[self.pos1])
        cell1 = board_state[self.pos1]

        piece, player = get_active_piece(cell1)
        new_cell1 = set_piece(cell1, piece, 0)

        #cell2 = byte_to_str(board_state[self.pos2])
        cell2 = board_state[self.pos2]
        new_cell2 = set_piece(cell2, piece, player_turn)

        new_board_state = np.array(board_state)
        #new_board_state[self.pos1] = str_to_byte(new_cell1)
        #new_board_state[self.pos2] = str_to_byte(new_cell2)
        new_board_state[self.pos1] = new_cell1
        new_board_state[self.pos2] = new_cell2

        new_hands = hands

        new_owns = np.array(owns)
        _,player = get_active_piece(new_cell1)
        new_owns[self.pos1//3,self.pos1%3] = -1 if player == 2 else player
        new_owns[self.pos2//3,self.pos2%3] = -1 if player_turn == 2 else player_turn

        return new_board_state, new_hands, new_owns

    def __repr__(self):
        return 'M <%d to %d>' % (self.pos1, self.pos2)

class PlacePiece:
    def __init__(self, piece_type, pos):
        self.piece_type = piece_type
        self.pos = pos

    def apply(self, board_state, player_turn, hands, owns):
        #cell = byte_to_str(board_state[self.pos])
        cell = board_state[self.pos]

        new_cell = ''
        for i in range(3):
            new_cell += cell[i] if i != self.piece_type else str(player_turn)

        #new_cell = str_to_byte(new_cell)
        new_cell = new_cell
        new_board_state = np.array(board_state)
        new_board_state[self.pos] = new_cell

        other_player = 1 if player_turn == 2 else 2
        hand = np.array(hands[player_turn])
        hand[self.piece_type] -= 1
        new_hands = {player_turn: hand, other_player: hands[other_player]}

        new_owns = np.array(owns)
        new_owns[self.pos//3,self.pos%3] = -1 if player_turn == 2 else player_turn

        return new_board_state, new_hands, new_owns
    
    def __repr__(self):
        return 'P <%d @ %d>' % (self.piece_type, self.pos)

class RandomGame:
    def __init__(self, game):
        self.game = game

    def play(self):
        valid_moves = self.game.get_valid_moves()
        random = np.random.randint(len(valid_moves))

        move = valid_moves[random]

        self.game = self.game.apply_move(move)

        print(move)
        print(self.game)

        if self.game.game_state() == -1:
            self.play()

class GameNode:
    def __init__(self, game):
        self.game = game
        self.children = {}

MAX_DEPTH = 10

class SolverRecursive:
    def __init__(self, game, max_depth=MAX_DEPTH):
        self.start_game = game
        self.max_depth = max_depth
        self.saved = {}
        self.solve_calls = 0
        self.breaks = 0
        self.stack = []

    def solve(self, game, depth=0, exec_path=[]):
        self.solve_calls += 1
        # 0 -> Draw, 1 -> Player 1 Victory, 2 -> Player 2 Victory, -1 -> None
        if game.game_state() != -1:
            ret = (game.game_state(), [], 0)
            self.saved[hash(game)] = ret

            return ret

        if depth >= self.max_depth:
            ret = (DRAW, [], 0)
            #self.saved[(hash(game),0)] = ret
            self.saved[hash(game)] = ret

            return ret

        player = game.turn
        other_player = 1 if player == 2 else 2

        if depth == 0:
            start_time = time.time()
            last_time = start_time

        valid_moves = game.get_valid_moves()

        results = np.zeros(len(valid_moves), dtype=np.int8)
        move_orders = []
        for i,move in enumerate(valid_moves):
            new_game = game.apply_move(move)
            hash_val = hash(new_game)

            # TODO: Compute BFS?
            depth_rem = self.max_depth - (depth+1)

            if hash_val in self.saved:
                result,move_order,depth_rem_saved = self.saved[hash(new_game)]

            if hash_val not in self.saved or (result == DRAW and depth_rem > depth_rem_saved):
                if hash_val in exec_path:
                    self.breaks += 1
                    result,move_order = DRAW, []
                else:
                    result,move_order,_ = self.solve(new_game, depth+1, exec_path+[hash_val])

            results[i] = result
            move_orders.append(move_order)

            if depth == 0:
                print('%d/%d (%.3f sec) -- %s' % (i+1, len(valid_moves), time.time()-last_time, move))
                last_time = time.time()

        if depth == 0:
            print('Total Time: %.3f sec' % (time.time()-start_time))
            print('Calls: %d' % self.solve_calls)
            print('Breaks: %d' % self.breaks)

        # Current player has winning path
        if np.any(results==player):
            for result,move,mo in zip(results,valid_moves,move_orders):
                if result==player:
                    hash_val,hash_rot = game.hash_rot() 
                    ret = (player, [(player,move,hash_rot)]+mo, 0)
                    self.saved[hash_val] = ret
                    
                    return ret

        # Other player has forced win
        if np.all(results==other_player):
            hash_val,hash_rot = game.hash_rot() 
            ret = (other_player, [(player,valid_moves[0],hash_rot)]+move_orders[0], 0)
            self.saved[hash_val] = ret

            return ret

        for result,move,mo in zip(results,valid_moves,move_orders):
            if result==DRAW:
                hash_val,hash_rot = game.hash_rot() 
                depth_rem = self.max_depth - depth
                ret = (DRAW, [(player,move,hash_rot)]+mo, depth_rem)
                #self.saved[(hash_val,self.max_depth-depth)] = ret
                self.saved[hash_val] = ret

                return ret

    def save(self, dest):
        print(len(self.saved))
        #self.saved = {k: v for (k,v) in self.saved.items() if v[0] != DRAW}
        #print(len(self.saved))
        print('Saving')
        with open(dest, 'wb') as fout:
            pickle.dump(self.saved, fout)

    def load(self, src):
        print('Loading')
        with open(src, 'rb') as fin:
            self.saved = pickle.load(fin)

class GameNode:
    def __init__(self, game):
        self.game = game
        self.children = []
        self.parents = []
        self.result = None

    def add_child(self, move, child):
        self.children.append((move,child))
        child.parents.append(self)

class Solver:
    def __init__(self, game, max_depth=MAX_DEPTH):
        self.start_game = game
        self.max_depth = max_depth
        self.saved = {}
        self.solve_calls = 0

    def build_tree(self, game):
        #stack = deque([])
        stack = []
        leaves = []
        root_game_node = GameNode(game)
        stack.append((0,root_game_node))

        while len(stack) > 0:
            #depth,game_node = stack.popleft()
            depth,game_node = stack.pop()

            game = game_node.game
            valid_moves = game.get_valid_moves()

            if depth==0:
                #count = 0
                #num_moves = len(valid_moves)
                cur_depth = 0
                last_time = time.time()

            if depth > cur_depth:
                cur_depth = depth
                now_time = time.time()
                print('%d/%d (%.3f sec)' % (depth, self.max_depth, now_time-last_time))
                last_time = now_time

            """
            if depth==1:
                count += 1
                now_time = time.time()
                print('%d/%d (%.3f sec)' % (count, num_moves, now_time-last_time))
                last_time = now_time
            """


            if valid_moves is None:
                leaves.append(game_node)

            for i,move in enumerate(valid_moves):
                new_game = game.apply_move(move)

                if hash(new_game) in self.saved:
                    new_game_node = self.saved[hash(new_game)]
                else:
                    new_game_node = GameNode(new_game)
                    self.saved[hash(new_game)] = new_game_node

                    if depth+1 >= self.max_depth:
                        leaves.append(new_game_node)
                    else:
                        stack.append((depth+1,new_game_node))

                game_node.add_child(move, new_game_node)

        now_time = time.time()
        print('%d/%d (%.3f sec)' % (depth+1, self.max_depth, now_time-last_time))

        return root_game_node, leaves

    def solve_tree(self, game_node, depth=0):
        self.solve_calls += 1

        game = game_node.game

        if game.game_state() != -1:
            ret = (game.game_state(), [])
            game_node.result = ret

            return ret

        if depth >= self.max_depth:
            ret = (DRAW, [])
            game_node = ret

            return ret

        player = game.turn
        other_player = 1 if player == 2 else 2

        if depth == 0:
            start_time = time.time()
            last_time = start_time

        results = np.zeros(len(game_node.children), dtype=np.int8)
        move_orders = []
        valid_moves = []
        for i,(move,child) in enumerate(game_node.children):
            if child.result:
                result,move_order = child.result
            else:
                result,move_order = self.solve_tree(child, depth+1)

            if depth == 0:
                print('%d/%d (%.3f sec)' % (i+1, len(game_node.children), time.time()-last_time))
                last_time = time.time()

            results[i] = result
            valid_moves.append(move)
            move_orders.append(move_order)

        if depth == 0:
            print('Total Time: %.3f sec' % (time.time()-start_time))
            print('Calls: %d' % self.solve_calls)

        # Current player has winning path
        if np.any(results==player):
            for result,move,mo in zip(results,valid_moves,move_orders):
                if result==player:
                    ret = (player, [(player,move)]+mo)
                    game_node.result = ret

                    return ret

        # Other player has forced win
        if np.all(results==other_player):
            print('t',game_node.game)
            print(depth)
            print(game_node.children)
            print(valid_moves, move_orders)
            ret = (other_player, [(player,valid_moves[0])]+move_orders[0])
            game_node.result = ret

            return ret

        for result,move,mo in zip(results,valid_moves,move_orders):
            if result==DRAW:
                ret = (DRAW, [(player,move)]+mo)
                game_node.result = ret

                return ret

    def solve(self, game):
        # 0 -> Draw, 1 -> Player 1 Victory, 2 -> Player 2 Victory, -1 -> None
        tree,leaves = self.build_tree(game)
        result,move_order = self.solve_tree(tree)

        return result, move_order
       
    def save(self, dest):
        """
        print(len(self.saved))
        self.saved = {k: v for (k,v) in self.saved.items() if v[0] != DRAW}
        print(len(self.saved))
        """
        print('Saving')
        with open(dest, 'wb') as fout:
            pickle.dump(self.saved, fout)

    def load(self, src):
        print('Loading')
        with open(src, 'rb') as fin:
            self.saved = pickle.load(fin)
