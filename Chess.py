# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import numpy as np
import time
from typing import *

class Chess:
    # Chess pieces
    PAWN = 0x1
    KNIGHT = 0x2
    BISHOP = 0x4
    ROOK = 0x8
    QUEEN = 0x10
    KING = 0x20
    PIECE = 0x3F
    # Color flags
    WHITE = 0x40
    BLACK = 0x80
    COLORS = 0xC0
    # All data
    PIECE_DATA = 0xFF
    # Extra data
    NOT_MOVED = 0x100
    
    # Movement type
    MOVE = 1
    LONGMOVE = 2
    MOVEKING = 3
    CASTLING = 4
    PROMOTION = 5
    ENPASSANT = 6
    
    def __init__(self, state = None):
        if state:
            self.state = state.state.copy()
            self.wking, self.bking = state.wking, state.bking
            self.turn = state.turn
            self.turn_counter = state.turn_counter
            self.enpassant = state.enpasse
            self.last_update = 0
        else:
            self.state = np.zeros((8,8), dtype=np.int16)
            self.reset()
    
    # Resets the board:
    def reset(self):
        # Clear board
        self.state[:] = 0
        # Set pawns
        self.state[1,:] = Chess.PAWN | Chess.WHITE
        self.state[6,:] = Chess.PAWN | Chess.BLACK
        # Set rooks
        self.state[0,0] = Chess.ROOK | Chess.WHITE | Chess.NOT_MOVED
        self.state[0,7] = Chess.ROOK | Chess.WHITE | Chess.NOT_MOVED
        self.state[7,0] = Chess.ROOK | Chess.BLACK | Chess.NOT_MOVED
        self.state[7,7] = Chess.ROOK | Chess.BLACK | Chess.NOT_MOVED
        # Set knights
        self.state[0,1] = Chess.KNIGHT | Chess.WHITE
        self.state[0,6] = Chess.KNIGHT | Chess.WHITE
        self.state[7,1] = Chess.KNIGHT | Chess.BLACK
        self.state[7,6] = Chess.KNIGHT | Chess.BLACK
        # Set bishops
        self.state[0,2] = Chess.BISHOP | Chess.WHITE
        self.state[0,5] = Chess.BISHOP | Chess.WHITE
        self.state[7,2] = Chess.BISHOP | Chess.BLACK
        self.state[7,5] = Chess.BISHOP | Chess.BLACK
        # Set queens
        self.state[0,3] = Chess.QUEEN | Chess.WHITE
        self.state[7,3] = Chess.QUEEN | Chess.BLACK
        # Set kings
        self.wking = (0,4)
        self.bking = (7,4)
        self.state[self.wking] = Chess.KING | Chess.WHITE | Chess.NOT_MOVED
        self.state[self.bking] = Chess.KING | Chess.BLACK | Chess.NOT_MOVED
        self.turn_counter = 1
        self.last_update = 0
        self.turn = Chess.WHITE
        self.enpassant = None
    
    # Sets turn related constants
    def set_turn(self):
        if self.turn_counter == self.last_update: return
        self.last_update = self.turn_counter
        if self.turn == Chess.WHITE:
            self.king = self.wking
            self.back_file = 0
            self.pawn_file = 1
            self.en_passant_file = 4
            self.end_file = 7
            self.dy = 1
            self.enemy_color = Chess.BLACK
        else:
            self.king = self.bking
            self.back_file = 7
            self.pawn_file = 6
            self.en_passant_file = 3
            self.end_file = 0
            self.dy = -1
            self.enemy_color = Chess.WHITE
        self.attacked_cells, self.fixed_cells, self.attacker_pieces = self.get_attacked(self.enemy_color)
    
    # Returns whether a position is a valid chess board position
    def valid_position(self, position : tuple):
        return position[0] >= 0 and position[0] < 8 and position[1] >= 0 and position[1] < 8
    
    # Check if a position is valid and if so adds it to set locations, return whether the piece is attacking the opposing king
    def add_valid_position(self, position : tuple, color : np.int8, locations : set):
        if self.valid_position(position):
            locations.add(position)
            if self.state[position] & Chess.KING and not self.state[position] & color: return True
        return False
    
    # Receives the location of a pawn and its color, adds the valid attacks to locations
    def get_attack_pawn(self, position : tuple, color : np.int8, locations : set, fixed : set = set(), attacker : set = set()):
        dy = 1 if color == Chess.WHITE else -1
        if self.add_valid_position((position[0]+dy, position[1]+1), color, locations) \
            or self.add_valid_position((position[0]+dy, position[1]-1), color, locations):
            attacker.add(position)
    
    # Receives the location of a knights and its color, adds the valid attacks to locations
    def get_attack_knight(self, position : tuple, color : np.int8, locations : set, fixed : set = set(), attacker : set = set()):
        if any(self.add_valid_position((position[0] + npos[0], position[1] + npos[1]), color, locations) for npos in ((2,1), (2,-1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2))):
            attacker.add(position)
    
    # Receives a genexpr of positions, adds the valid ones including the first hit, continues to check if the piece hit is fixed or if it is a check
    def get_attack_pierce(self, position : tuple, color : np.int8, locations : set, fixed : set, attacker : set, genexpr):
        fixpoint = None
        hit = False
        for npos in genexpr:
            if not hit:
                locations.add(npos)
            if self.state[npos]:
                if self.state[npos] & color: break
                if hit:
                    if self.state[npos] & Chess.KING: fixed.add(fixpoint)
                    break
                else:
                    hit, fixpoint = True, npos
                    if self.state[npos] & Chess.KING:
                        attacker.add(position)
                        break
    
    # Receives the location of a bishop and its color, adds the valid attacks to locations, if any piece is fixed adds it to the fixed list
    def get_attack_bishop(self, position : tuple, color : np.int8, locations : set, fixed : set = set(), attacker : set = set()):
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0]-i, position[1]-i) for i in range(1, min(position)+1)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0]-i, position[1]+i) for i in range(1, min(position[0], 7-position[1])+1)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0]+i, position[1]-i) for i in range(1, min(7-position[0], position[1])+1)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0]+i, position[1]+i) for i in range(1, min(7-position[0], 7-position[1])+1)))
    
    # Receives the location of a rook and its color, adds the valid attacks to locations, if any piece is fixed adds it to the fixed list
    def get_attack_rook(self, position : tuple, color : np.int8, locations : set, fixed : set = set(), attacker : set = set()):
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((i, position[1]) for i in range(position[0]-1,-1,-1)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0], i) for i in range(position[1]-1,-1,-1)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((i, position[1]) for i in range(position[0]+1, 8)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0], i) for i in range(position[1]+1, 8)))
    
    # Receives the location of a queen and its color, adds the valid attacks to locations
    def get_attack_queen(self, position : tuple, color : np.int8, locations : set, fixed : set = set(), attacker : set = set()):
        self.get_attack_bishop(position, color, locations, fixed, attacker)
        self.get_attack_rook(position, color, locations, fixed, attacker)


    # Receives the location of a king and its color, adds the valid attacks to locations
    def get_attack_king(self, position : tuple, color : np.int8, locations : set, fixed : set = set(), attacker : set = set()):
        for pos in ((position[0]-1,position[1]-1), (position[0]-1,position[1]), (position[0]-1,position[1]+1),
                    (position[0],position[1]-1), (position[0],position[1]+1),
                    (position[0]+1,position[1]-1), (position[0]+1,position[1]), (position[0]+1,position[1]+1)):
            if self.valid_position(pos) and not self.state[pos] & color: locations.add(pos)
            
    # Returns the set of all locations being attacked by color, the positions of all the opponent's fixed pieces and all pieces currently attacking the opposing king
    def get_attacked(self, color : np.int8) -> set:
        locations, fixed, attacker = set(), set(), set()
        for pos in ((i,j) for i in range(8) for j in range(8)):
            if self.state[pos] & color:
                if self.state[pos] & Chess.PAWN:
                    self.get_attack_pawn(pos, color, locations, fixed, attacker)
                elif self.state[pos] & Chess.KNIGHT:
                    self.get_attack_knight(pos, color, locations, fixed, attacker)
                elif self.state[pos] & Chess.BISHOP:
                    self.get_attack_bishop(pos, color, locations, fixed, attacker)
                elif self.state[pos] & Chess.ROOK:
                    self.get_attack_rook(pos, color, locations, fixed, attacker)
                elif self.state[pos] & Chess.QUEEN:
                    self.get_attack_queen(pos, color, locations, fixed, attacker)
                else:
                    self.get_attack_king(pos, color, locations, fixed, attacker)
        return locations,fixed,attacker
    
    # Returns the set of all locations that can be blocked to negate the current attacker (only works when there is a single attacker)
    def get_block_attacker(self) -> set:
        locations = set()
        for position in self.attacker_pieces:
            locations.add(position)
            if not (self.state[position] & (Chess.PAWN | Chess.KNIGHT)):
                dy = 1 if position[0] < self.king[0] else -1
                dx = 1 if position[1] < self.king[1] else -1
                if (self.state[position] & Chess.BISHOP) or ((self.state[position] & Chess.QUEEN) and (position[0] != self.king[0] and position[1] != self.king[1])):
                    for i in range(abs(position[0] - self.king[0])):
                        locations.add((position[0] + dy * i, position[1] + dx * i))
                if (self.state[position] & Chess.ROOK) or ((self.state[position] & Chess.QUEEN) and (position[0] == self.king[0] or position[1] == self.king[1])):
                    if position[0] == self.king[0]:
                        for i in range(abs(position[1] - self.king[1])):
                            locations.add((position[0], position[1] + dx * i))
                    else:
                        for i in range(abs(position[0] - self.king[0])):
                            locations.add((position[0] + dy * i, position[1]))
        return locations
    
    # Adds the possible action tuples to a list of actions for a given pawn
    def add_available_actions_pawn(self, position : tuple, actions : list):
        y,x = position
        # Gets available pawn attack options, if would arrive on endfile then promote
        for dx in (-1,1):
            npos = (y+self.dy, x+dx)
            if self.valid_position(npos) and self.state[npos] & self.enemy_color:
                if npos[0] != self.end_file:
                    actions.append( (Chess.MOVE, position, npos) )
                else:
                    for promotion in (Chess.KNIGHT, Chess.QUEEN): actions.append( (Chess.PROMOTION, position, npos, promotion) )
        # Gets en passant attack option
        if self.enpassant and y == self.en_passant_file and abs(x-self.enpassant[1])==1:
            actions.append( (Chess.ENPASSANT, position, self.enpassant) )
        # Gets pawn move options
        if y + self.dy != self.end_file:
            if not self.state[y+self.dy, x]:
                actions.append( (Chess.MOVE, position, (y+self.dy, x)) )
                # Gets pawn long move for initial line
                if (y == self.pawn_file and not self.state[y+(2*self.dy), x]):
                    actions.append( (Chess.LONGMOVE, position, (y+(2*self.dy), x)) )
        else:
            # Promotion of pawns
            for promotion in (Chess.KNIGHT, Chess.QUEEN): actions.append( (Chess.PROMOTION, position, npos, promotion) )
    
    # Adds the possible action tuples to a list of actions for a given warrior piece (knight/bishop/rook/queen)
    def add_available_actions_warrior(self, position : tuple, actions : list):
        locations = set()
        if self.state[position] & Chess.KNIGHT:
            self.get_attack_knight(position, self.turn, locations)
        elif self.state[position] & Chess.BISHOP:
            self.get_attack_bishop(position, self.turn, locations)
        elif self.state[position] & Chess.ROOK:
            self.get_attack_rook(position, self.turn, locations)
        elif self.state[position] & Chess.QUEEN:
            self.get_attack_queen(position, self.turn, locations)
        for npos in locations: 
            if not self.state[npos] & self.turn:
                actions.append( (Chess.MOVE, position, npos) )
    
    # Adds the possible action tuples to a list of actions for the king
    def add_available_actions_king(self, position : tuple, actions : list):
        locations = set()
        self.get_attack_king(position, self.turn, locations)
        for npos in locations:
            if not self.state[npos] & self.turn and not npos in self.attacked_cells and not npos in self.fixed_cells:
                actions.append( (Chess.MOVEKING, position, npos) )
        # Get special castling move: King nor tower has moved, spaces are empty, king is not on check, nor ending positions for tower and king are on check
        if self.state[position] & Chess.NOT_MOVED and not self.attacker_pieces:
            if self.state[self.back_file, 0] & Chess.NOT_MOVED and np.sum(self.state[self.back_file, 1:4]) == 0 and not (self.back_file, 2) in self.attacked_cells and not (self.back_file, 3) in self.attacked_cells:
                actions.append( (Chess.CASTLING, position, (self.back_file, 2)) )
            if self.state[self.back_file, 7] & Chess.NOT_MOVED and np.sum(self.state[self.back_file, 5:7]) == 0 and not (self.back_file, 5) in self.attacked_cells and not (self.back_file, 6) in self.attacked_cells:
                actions.append( (Chess.CASTLING, position, (self.back_file, 6)) )
                
    # Returns a list of actions available to the player
    def get_available_actions(self):
        self.set_turn()
        actions = []
        # If more than 1 piece is attacking the king at once, it is useless to block, only the king may move
        if len(self.attacker_pieces) > 1:
            print("King under double attack")
            self.add_available_actions_king(self.king, actions)
            return actions
        for pos in ((i,j) for i in range(8) for j in range(8)):
            if self.state[pos] & self.turn and not pos in self.fixed_cells:
                if self.state[pos] & Chess.PAWN:
                    self.add_available_actions_pawn(pos, actions)
                elif self.state[pos] & Chess.KING:
                    self.add_available_actions_king(pos, actions)
                else:
                    self.add_available_actions_warrior(pos, actions)
        # If king is checked by a single piece then can move to defend or block attack only
        if self.attacker_pieces:
            locations = self.get_block_attacker()
            filtered_actions = []
            for action in actions:
                act, pos, npos = action
                if npos in locations or (self.state[pos] & Chess.KING):
                    filtered_actions.append(action)
            actions = filtered_actions
        return actions
    
    # Changes the state of the chess board according to the action
    def do_action(self, action : tuple, check : bool = False):
        self.set_turn()
        if check and not action in self.get_available_actions():
            print("Available actions:", self.get_available_actions())
            print("Attacked:", self.set_as_notation(self.attacked_cells))
            print("Fixed:", self.set_as_notation(self.fixed_cells))
            print("Is king attacked?", self.king in self.attacked_cells)
            raise Exception("Invalid action: " + str(action))
        act, pos, npos = action[0], action[1], action[2]
        self.enpassant = None
        if act == Chess.MOVE:
            self.state[npos], self.state[pos] = self.state[pos] & Chess.PIECE_DATA, 0
        elif act == Chess.LONGMOVE:
            self.state[npos], self.state[pos] = self.state[pos], 0
            self.enpassant = ((pos[0] + npos[0])//2, pos[1])
        elif act == Chess.MOVEKING:
            self.state[npos], self.state[pos] = self.state[pos] & Chess.PIECE_DATA, 0
            if self.turn == Chess.WHITE:
                self.wking = npos
            else:
                self.bking = npos
        elif act == Chess.CASTLING:
            self.state[npos], self.state[pos] = self.state[pos] & Chess.PIECE_DATA, 0
            if pos[1] < npos[1]:
                self.state[(pos[0], 5)], self.state[(pos[0], 7)] = self.state[(pos[0], 7)] & Chess.PIECE_DATA, 0
            else:
                self.state[(pos[0], 3)], self.state[(pos[0], 0)] = self.state[(pos[0], 0)] & Chess.PIECE_DATA, 0
            if self.turn == Chess.WHITE:
                self.wking = npos
            else:
                self.bking = npos
        elif act == Chess.PROMOTION:
            self.state[npos], self.state[pos] = (self.state[pos] & Chess.COLORS) | action[3], 0
        elif act == Chess.ENPASSANT:
            self.state[npos], self.state[pos] = self.state[pos], 0
            self.state[(pos[0], npos[1])] = 0
        else:
            raise Exception("Unknown action taken: " + str(action))
        self.turn = Chess.BLACK if self.turn == Chess.WHITE else Chess.WHITE
        self.turn_counter += 1
        
    # Changes the state of the chess board according to the action given in algebraic notation
    def do_action_algebraic(self, action : str, check : bool = False):
        self.set_turn()
        if action.startswith('0') or action.startswith('O'):
            back_line = 0 if self.turn == Chess.WHITE else 7
            pos = (back_line, 4)
            npos = (back_line, 2 if action.startswith('0-0-0') or action.startswith('O-O-O') else 6)
            self.do_action((Chess.CASTLING, pos, npos), check = check)
            return
        pieces = {'N': Chess.KNIGHT, 'B': Chess.BISHOP, 'R': Chess.ROOK, 'Q': Chess.QUEEN, 'K': Chess.KING}
        piece = pieces.get(action[0], Chess.PAWN)
        promotion = None
        letters = []
        numbers = []
        for i,c in enumerate(action):
            if c >= '1' and c <= '8':
                numbers.append(ord(c) - ord('1'))
            elif c >= 'a' and c <= 'h':
                letters.append(ord(c) - ord('a'))
            elif c in pieces and i:
                promotion = pieces.get(c)
        npos = (numbers[-1], letters[-1])
        row, column = numbers[0] if len(numbers) > 1 else None, letters[0] if len(letters)>1 else None
        real_action = None
        for pos in ((i,j) for i in range(8) for j in range(8)):
            if row and row != pos[0]: continue
            if column and column != pos[1]: continue
            if not (self.state[pos] & piece) or not (self.state[pos] & self.turn) or (pos in self.fixed_cells): continue
            actions = []
            if piece == Chess.PAWN:
                self.add_available_actions_pawn(pos, actions)
            elif piece == Chess.KING:
                self.add_available_actions_king(pos, actions)
            else:
                self.add_available_actions_warrior(pos, actions)
            if promotion != piece:
                for act in actions:
                    if act[2] == npos: real_action = act; break
            else:
                for act in actions:
                    if act[2] == npos: real_action = (act[0], act[1], act[2], promotion); break
            if real_action: break
        if real_action:
            self.do_action(real_action, check)
            return
        raise Exception("Could not handle action: " + action)
    
    # Returns a tuple (has_ended, winner) which contains whether the game has ended and which the winner is 'W'/'D'/'B'
    def has_ended(self):
        actions = self.get_available_actions()
        if actions:
            return (False, '')
        else:
            if self.attacker_pieces:
                return (True, 'B' if self.turn == Chess.WHITE else 'W')
            else:
                return (True, 'D')
    
    # Returns a board position in chess notation
    def pos_as_notation(self, position : tuple) -> str:
        return chr(position[1] + ord('A')) + chr(position[0] + ord('1'))
    
    def set_as_notation(self, positions : set) -> str:
        arr = [self.pos_as_notation(pos) for pos in positions]
        arr.sort()
        return ' '.join(arr)
    
    # Returns board in text representation for printing in console
    def __str__(self):
        chars = {
            Chess.BLACK: u'\u25FB',
            Chess.BLACK | Chess.PAWN: u'\u265F',
            Chess.BLACK | Chess.ROOK: u'\u265C',
            Chess.BLACK | Chess.KNIGHT: u'\u265E',
            Chess.BLACK | Chess.BISHOP: u'\u265D',
            Chess.BLACK | Chess.KING: u'\u265A',
            Chess.BLACK | Chess.QUEEN: u'\u265B',
            Chess.WHITE: u'\u25FC',
            Chess.WHITE | Chess.PAWN: u'\u2659',
            Chess.WHITE | Chess.ROOK: u'\u2656',
            Chess.WHITE | Chess.KNIGHT: u'\u2658',
            Chess.WHITE | Chess.BISHOP: u'\u2657',
            Chess.WHITE | Chess.KING: u'\u2654',
            Chess.WHITE | Chess.QUEEN: u'\u2655'
        }
        board = [[chars[Chess.BLACK], chars[Chess.WHITE]]*4 if i%2 else [chars[Chess.WHITE], chars[Chess.BLACK]]*4 for i in range(8)]
        for pos in ((i,j) for i in range(8) for j in range(8)):
            if self.state[pos]:
                board[pos[0]][pos[1]] = chars[self.state[pos] & (Chess.COLORS | Chess.PIECE)]
        board = board[::-1]
        s = ''
        for i in range(8):
            s += '{:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3}\n'.format(*board[i])
        return s
    
if __name__ == "__main__":
    actions = 'e4 e5 Nf3 Nc6 Bc4 Bc5 c3 d6 d4 exd4 cxd4 Bb4+ Nc3 Nf6 O-O Bxc3 bxc3 Nxe4 Re1 d5 Rxe4+ dxe4 Ng5 O-O Qh5 h6 Nxf7 Qf6 Nxh6+ Kh8 Nf7+ Kg8 Qh8#'.split(' ')
    game = Chess()
    print("State:\n" + str(game))
    for act in actions:
        print("Taking action:", act, "by",  ('W' if game.turn == Chess.WHITE else 'B'))
        game.do_action_algebraic(act, check=True)
        print("State:\n" + str(game))
    print("End state:", game.has_ended())
    
