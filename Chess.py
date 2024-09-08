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
    PROMOTION_KNIGHT = 5
    PROMOTION_QUEEN = 6
    ENPASSANT = 7
    
    def __init__(self, state = None):
        if state:
            self.state = state.state.copy()
            self.turn = state.turn
            self.enpassant = state.enpasse
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
        self.turn = Chess.WHITE
        self.enpassant = None
    
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
    
    # Receives the location of a pawn and the position of an attacked king, adds the valid locations to block the attack
    def get_block_pawn(self, position : tuple, king : tuple, locations : set):
        locations.add(position)
    
    # Receives the location of a knights and its color, adds the valid attacks to locations
    def get_attack_knight(self, position : tuple, color : np.int8, locations : set, fixed : set = set(), attacker : set = set()):
        if any(self.add_valid_position(npos, color, locations) for npos in ((2,1), (2,-1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2))):
            attacker.add(position)
    
    # Receives the location of a knight and the position of an attacked king, adds the valid locations to block the attack
    def get_block_knight(self, position : tuple, king : tuple, locations : set):
        locations.add(position)
    
    # Receives a genexpr of positions, adds the valid ones including the first hit, continues to check if the piece hit is fixed or if it is a check
    def get_attack_pierce(self, position : tuple, color : np.int8, locations : set, fixed : set, attacker : set, genexpr):
        fixpoint = None
        hit = False
        for npos in genexpr:
            if not hit:
                locations.add(npos)
            if self.state[npos]:
                if self.state[npos] & color: break
                if self.state[npos] & Chess.KING: attacker.add(position); break
                if hit:
                    if self.state[npos] & Chess.KING: fixed.add(fixpoint); break
                else:
                    hit, fixpoint = True, npos
    
    # Receives the location of a bishop and its color, adds the valid attacks to locations, if any piece is fixed adds it to the fixed list
    def get_attack_bishop(self, position : tuple, color : np.int8, locations : set, fixed : set = set(), attacker : set = set()):
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0]-i, position[1]-i) for i in range(1, min(position)+1)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0]-i, position[1]+i) for i in range(1, min(position[0], 7-position[1])+1)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0]+i, position[1]-i) for i in range(1, min(7-position[0], position[1])+1)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0]+i, position[1]+i) for i in range(1, min(7-position[0], 7-position[1])+1)))
    
    # Receives the location of a bishop and the position of an attacked king, adds the valid locations to block the attack
    def get_block_bishop(self, position : tuple, king : tuple, locations : set):
        dy = 1 if position[0] < king[0] else -1
        dx = 1 if position[1] < king[1] else -1
        for i in range(abs(position[0] - king[0])):
            locations.add((position[0] + dy * i, position[1] + dx * i))

    # Receives the location of a rook and its color, adds the valid attacks to locations, if any piece is fixed adds it to the fixed list
    def get_attack_rook(self, position : tuple, color : np.int8, locations : set, fixed : set = set(), attacker : set = set()):
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((i, position[1]) for i in range(position[0]-1,-1,-1)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0], i) for i in range(position[1]-1,-1,-1)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((i, position[1]) for i in range(position[0]+1, 8)))
        self.get_attack_pierce(position, color, locations, fixed, attacker, ((position[0], i) for i in range(position[1]+1, 8)))
    
    # Receives the location of a rook and the position of an attacked king, adds the valid locations to block the attack
    def get_block_rook(self, position : tuple, king : tuple, locations : set):
        if position[0] == king[0]:
            dx = 1 if position[1] < king[1] else -1
            for i in range(abs(position[1] - king[1])):
                locations.add((position[0], position[1] + dx * i))
        else:
            dy = 1 if position[0] < king[0] else -1
            for i in range(abs(position[0] - king[0])):
                locations.add((position[0] + dy * i, position[1]))
        
    
    # Receives the location of a queen and its color, adds the valid attacks to locations
    def get_attack_queen(self, position : tuple, color : np.int8, locations : set, fixed : set = set(), attacker : set = set()):
        self.get_attack_bishop(position, color, locations, fixed, attacker)
        self.get_attack_rook(position, color, locations, fixed, attacker)
        
    # Receives the location of a queen and the position of an attacked king, adds the valid locations to block the attack
    def get_block_queen(self, position : tuple, king : tuple, locations : set):
        if position[0] == king[0] or position[1] == king[1]:
            self.get_block_rook(position, king, locations)
        else:
            self.get_block_bishop(position, king, locations)
    
    # Receives the location of a king and its color, adds the valid attacks to locations
    def get_attack_king(self, position : tuple, color : np.int8, locations : set, fixed : set = set(), attacker : set = set()):
        for pos in ((position[0]-1,position[1]-1), (position[0]-1,position[1]), (position[0]-1,position[1]+1),
                    (position[0],position[1]-1), (position[0],position[1]+1),
                    (position[0]+1,position[1]-1), (position[0]+1,position[1]), (position[0]+1,position[1]+1)):
            if self.valid_position(pos) and not self.state[pos] & color: locations.add(pos)
    
    # Returns the set of all locations being attacked by color and the positions of fixed pieces
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
    
    # Returns a list of actions available to the player
    def get_available_actions(self):
        actions = []
        king = self.wking if self.turn == Chess.WHITE else self.bking
        back_file,pawn_file,en_passant_file,end_file,dy,enemy_color = (0,1,4,7,1,Chess.BLACK) if self.turn == Chess.WHITE else (7,6,3,0,-1,Chess.WHITE)
        attacked_cells,fixed_cells,attacker_pieces = self.get_attacked(enemy_color)
        # If more than 1 piece is attacking the king at once, it is useless to block, only the king may move
        if len(attacker_pieces) > 1:
            # Gets available king actions
            locations = set()
            self.get_attack_king(king, self.turn, locations)
            for npos in locations: 
                if not self.state[npos] & self.turn and not npos in attacked_cells and not npos in fixed_cells:
                    actions.append( (Chess.MOVEKING, king, npos) )
            return actions
        for pos in ((i,j) for i in range(8) for j in range(8)):
            if self.state[pos] & self.turn and not pos in fixed_cells:
                locations = set()
                if self.state[pos] & Chess.PAWN:
                    # Gets available pawn attack options, if would arrive on endfile then promote
                    if pos[1] > 0 and self.state[(pos[0]+dy, pos[1]-1)] & enemy_color:
                        if pos[0]+dy != end_file:
                            actions.append( (Chess.MOVE, pos, (pos[0]+dy, pos[1]-1)) )
                        else:
                            for promotion in (Chess.PROMOTION_KNIGHT, Chess.PROMOTION_QUEEN): actions.append( (promotion, pos, (pos[0]+dy, pos[1]-1)) )
                    if pos[1] < 7 and self.state[(pos[0]+dy, pos[1]+1)] & enemy_color:
                        if pos[0]+dy != end_file:
                            actions.append( (Chess.MOVE, pos, (pos[0]+dy, pos[1]+1)) )
                        else:
                            for promotion in (Chess.PROMOTION_KNIGHT, Chess.PROMOTION_QUEEN): actions.append( (promotion, pos, (pos[0]+dy, pos[1]+1)) )
                    # Gets en passant attack option
                    if self.enpassant and pos[0] == en_passant_file and abs(pos[1]-self.enpassant[1])==1:
                        actions.append( (Chess.ENPASSANT, pos, self.enpassant) )
                    # Gets pawn move options
                    if pos[0] != end_file:
                        if not self.state[pos[0]+dy, pos[1]]:
                            actions.append( (Chess.MOVE, pos, (pos[0]+dy, pos[1])) )
                            # Gets pawn long move for initial line
                            if (pos[0] == pawn_file and not self.state[pos[0]+(2*dy), pos[1]]):
                                actions.append( (Chess.LONGMOVE, pos, (pos[0]+(2*dy), pos[1])) )
                    else:
                        # Promotion of pawns
                        for promotion in (Chess.PROMOTION_KNIGHT, Chess.PROMOTION_QUEEN): actions.append( (promotion, pos, (pos[0]+dy, pos[1])) )
                elif self.state[pos] & Chess.KNIGHT:
                    # Gets available knight actions
                    self.get_attack_knight(pos, self.turn, locations)
                    for npos in locations: 
                        if not self.state[npos] & self.turn:
                            actions.append( (Chess.MOVE, pos, npos) )
                elif self.state[pos] & Chess.BISHOP:
                    # Gets available bishop actions
                    self.get_attack_bishop(pos, self.turn, locations)
                    for npos in locations: 
                        if not self.state[npos] & self.turn:
                            actions.append( (Chess.MOVE, pos, npos) )
                elif self.state[pos] & Chess.ROOK:
                    # Gets available rook actions
                    self.get_attack_rook(pos, self.turn, locations)
                    for npos in locations: 
                        if not self.state[npos] & self.turn:
                            actions.append( (Chess.MOVE, pos, npos) )
                elif self.state[pos] & Chess.QUEEN:
                    # Gets available queen actions
                    self.get_attack_queen(pos, self.turn, locations)
                    for npos in locations: 
                        if not self.state[npos] & self.turn:
                            actions.append( (Chess.MOVE, pos, npos) )
                else:
                    # Gets available king actions
                    self.get_attack_king(pos, self.turn, locations)
                    for npos in locations: 
                        if not self.state[npos] & self.turn and not npos in attacked_cells and not npos in fixed_cells:
                            actions.append( (Chess.MOVEKING, pos, npos) )
                    # Get special castling move: King nor tower has moved, spaces are empty, king is not on check, nor ending positions for tower and king are on check
                    if self.state[pos] & Chess.NOT_MOVED and not attacker_pieces:
                        if self.state[back_file, 0] & Chess.NOT_MOVED and np.sum(self.state[back_file, 1:4]) == 0 and not (back_file, 1) in attacked_cells and not (back_file, 2) in attacked_cells:
                            actions.append( (Chess.CASTLING, pos, (back_file, 1)) )
                        if self.state[back_file, 7] & Chess.NOT_MOVED and np.sum(self.state[back_file, 5:7]) == 0 and not (back_file, 5) in attacked_cells and not (back_file, 6) in attacked_cells:
                            actions.append( (Chess.CASTLING, pos, (back_file, 6)) )
        # If king is checked by a single piece then can move to defend or block attack only
        if attacker_pieces:
            locations = set()
            for pos in attacker_pieces:
                if self.state[pos] & Chess.PAWN:
                    self.get_block_pawn(pos, king, locations)
                elif self.state[pos] & Chess.KNIGHT:
                    self.get_block_knight(pos, king, locations)
                elif self.state[pos] & Chess.BISHOP:
                    self.get_block_bishop(pos, king, locations)
                elif self.state[pos] & Chess.ROOK:
                    self.get_block_rook(pos, king, locations)
                elif self.state[pos] & Chess.QUEEN:
                    self.get_block_queen(pos, king, locations)
                else:
                    raise Exception("A king should never be attacking another king, something weird happened")
            filtered_actions = []
            for action in actions:
                act, pos, npos = action
                if npos in locations and (not self.state[pos] & Chess.KING or len(locations) == 1):
                    filtered_actions.append(action)
            actions = filtered_actions
        return actions
    
    # Changes the state of the chess board according to the action
    def do_action(self, action, check=False):
        if check and not action in self.get_available_actions():
            raise Exception("Invalid action: " + str(action))
        act, pos, npos = action
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
            pass
        elif act == Chess.PROMOTION_KNIGHT:
            self.state[npos], self.state[pos] = (self.state[pos] & Chess.COLORS) | Chess.KNIGHT, 0
        elif act == Chess.PROMOTION_QUEEN:
            self.state[npos], self.state[pos] = (self.state[pos] & Chess.COLORS) | Chess.QUEEN, 0
        elif act == Chess.ENPASSANT:
            self.state[npos], self.state[pos] = self.state[pos], 0
            self.state[(pos[0], npos[1])] = 0
        else:
            raise Exception("Unknown action taken: " + str(action))
        self.turn = Chess.BLACK if self.turn == Chess.WHITE else Chess.WHITE
    
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
        
c = Chess()
t = time.time()
for i in range(1000):
    actions = c.get_available_actions()
print("Time:", time.time()- t)
print(actions)

print(c)

