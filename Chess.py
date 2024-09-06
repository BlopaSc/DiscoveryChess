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
    # Extra data
    MOVED = 0x100
    
    # Movement type
    MOVE = 1
    LONGMOVE = 2
    CASTLING = 3
    PROMOTION_KNIGHT = 4
    PROMOTION_QUEEN = 5
    ENPASSANT = 6
    
    def __init__(self, state = None):
        if state:
            self.state = state.state.copy()
            self.turn = state.turn
            self.wking = state.wking
            self.bking = state.bking
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
        self.state[0,0] = Chess.ROOK | Chess.WHITE
        self.state[0,7] = Chess.ROOK | Chess.WHITE
        self.state[7,0] = Chess.ROOK | Chess.BLACK
        self.state[7,7] = Chess.ROOK | Chess.BLACK
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
        self.state[self.wking] = Chess.KING | Chess.WHITE
        self.state[self.bking] = Chess.KING | Chess.BLACK
        self.turn = Chess.WHITE
        self.enpassant = None
    
    # Returns whether a position is a valid chess board position
    def valid_position(self, position : tuple):
        return position[0] >= 0 and position[0] < 8 and position[1] >= 0 and position[1] < 8
    
    # Check if a position is valid and if so adds it to set locations
    def add_valid_position(self, position : tuple, locations : set):
        if self.valid_position(position):
            locations.add(position)
    
    # Receives the location of a pawn and its color, adds the valid attacks to locations
    def get_attack_pawn(self, position : tuple, color : np.int8, locations : set):
        dy = 1 if color == Chess.WHITE else -1
        self.add_valid_position((position[0]+dy, position[1]+1), locations)
        self.add_valid_position((position[0]+dy, position[1]-1), locations)
    
    # Receives the location of a knights and its color, adds the valid attacks to locations
    def get_attack_knight(self, position : tuple, color : np.int8, locations : set):
        self.add_valid_position((position[0]+2, position[1]+1), locations)
        self.add_valid_position((position[0]+2, position[1]-1), locations)
        self.add_valid_position((position[0]-2, position[1]+1), locations)
        self.add_valid_position((position[0]-2, position[1]-1), locations)
        self.add_valid_position((position[0]+1, position[1]+2), locations)
        self.add_valid_position((position[0]-1, position[1]+2), locations)
        self.add_valid_position((position[0]+1, position[1]-2), locations)
        self.add_valid_position((position[0]-1, position[1]-2), locations)
        
    # Receives the location of a bishop and its color, adds the valid attacks to locations
    def get_attack_bishop(self, position : tuple, color : np.int8, locations : set):
        for i in range(1, min(position)+1):
            npos = (position[0]-i, position[1]-i)
            if not self.state[npos]:
                locations.add(npos)
            else:
                if not (self.state[npos] & color): locations.add(npos)
                break
        for i in range(1, min(position[0], 7-position[1])+1):
            npos = (position[0]-i, position[1]+i)
            if not self.state[npos]:
                locations.add(npos)
            else:
                if not (self.state[npos] & color): locations.add(npos)
                break
        for i in range(1, min(7-position[0], position[1])+1):
            npos = (position[0]+i, position[1]-i)
            if not self.state[npos]:
                locations.add(npos)
            else:
                if not (self.state[npos] & color): locations.add(npos)
                break
        for i in range(1, min(7-position[0], 7-position[1])+1):
            npos = (position[0]+i, position[1]+i)
            if not self.state[npos]:
                locations.add(npos)
            else:
                if not (self.state[npos] & color): locations.add(npos)
                break
    
    # Receives the location of a rook and its color, adds the valid attacks to locations
    def get_attack_rook(self, position : tuple, color : np.int8, locations : set):
        for i in range(position[0]-1,-1,-1):
            npos = (i, position[1])
            if not self.state[npos]:
                locations.add(npos)
            else:
                if not (self.state[npos] & color): locations.add(npos)
                break
        for i in range(position[1]-1,-1,-1):
            npos = (position[0], i)
            if not self.state[npos]:
                locations.add(npos)
            else:
                if not (self.state[npos] & color): locations.add(npos)
                break
        for i in range(position[0]+1, 8):
            npos = (i, position[1])
            if not self.state[npos]:
                locations.add(npos)
            else:
                if not (self.state[npos] & color): locations.add(npos)
                break
        for i in range(position[1]+1, 8):
            npos = (position[0], i)
            if not self.state[npos]:
                locations.add(npos)
            else:
                if not (self.state[npos] & color): locations.add(npos)
                break
    
    # Receives the location of a queen and its color, adds the valid attacks to locations
    def get_attack_queen(self, position : tuple, color : np.int8, locations : set):
        self.get_attack_bishop(position, color, locations)
        self.get_attack_rook(position, color, locations)
    
    # Receives the location of a king and its color, adds the valid attacks to locations
    def get_attack_king(self, position : tuple, color : np.int8, locations : set):
        for pos in ((position[0]-1,position[1]-1), (position[0]-1,position[1]), (position[0]-1,position[1]+1),
                    (position[0],position[1]-1), (position[0],position[1]+1),
                    (position[0]+1,position[1]-1), (position[0]+1,position[1]), (position[0]+1,position[1]+1)):
            if self.valid_position(pos) and not self.state[pos] & color: locations.add(pos)
    
    # Returns whether a position is currently attacked by color
    def is_attacked(self, position : tuple, color : np.int8) -> bool:
        locations = set()
        for pos in ((i,j) for i in range(8) for j in range(8)):
            if self.state[pos] & color:
                if self.state[pos] & Chess.PAWN:
                    self.get_attack_pawn(pos, color, locations)
                elif self.state[pos] & Chess.KNIGHT:
                    self.get_attack_knight(pos, color, locations)
                elif self.state[pos] & Chess.BISHOP:
                    self.get_attack_bishop(pos, color, locations)
                elif self.state[pos] & Chess.ROOK:
                    self.get_attack_rook(pos, color, locations)
                elif self.state[pos] & Chess.QUEEN:
                    self.get_attack_queen(pos, color, locations)
                else:
                    self.get_attack_king(pos, color, locations)
                if position in locations:
                    return True
        return False
    
    # Returns a dictionary of actions available to the player
    def get_available_actions(self):
        actions = []
        king = self.wking if self.turn == Chess.WHITE else self.bking
        init_line,dy,enemy = (1,1,Chess.BLACK) if self.turn == Chess.WHITE else (6,-1,Chess.WHITE)
        for pos in ((i,j) for i in range(8) for j in range(8)):
            if self.state[pos] & self.turn:
                locations = set()
                if self.state[pos] & Chess.PAWN:
                    # Gets available pawn attack options
                    if pos[1] > 0 and self.state[(pos[0], pos[1]-1)] & enemy:
                        actions.append( (Chess.MOVE, pos, (pos[0], pos[1]-1)) )
                    if pos[1] < 7 and self.state[(pos[0], pos[1]+1)] & enemy:
                        actions.append( (Chess.MOVE, pos, (pos[0], pos[1]+1)) )
                    # TODO: Add enpassant pawn attack option
                    # Gets pawn move options
                    if pos[0] < 7:
                        if not self.state[pos[0]+dy, pos[1]]:
                            actions.append( (Chess.MOVE, pos, (pos[0]+dy, pos[1])) )
                            # Gets pawn long move for initial line
                            if (pos[0] == init_line and not self.state[pos[0]+(2*dy), pos[1]]):
                                actions.append( (Chess.LONGMOVE, pos, (pos[0]+(2*dy), pos[1])) )
                    else:
                        # Promotion of pawns
                        actions.append( (Chess.PROMOTION_KNIGHT, pos, (pos[0]+dy, pos[1])) )
                        actions.append( (Chess.PROMOTION_QUEEN, pos, (pos[0]+dy, pos[1])) )
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
                        if not self.state[npos] & self.turn:
                            actions.append( (Chess.MOVE, pos, npos) )
                    # TODO: Add castling special move
        # TODO: Check if valid action? i.e: an action that leaves the king in a checked position
        return actions
    
    # TODO: Implement do action
    def do_action(self, action):
        if action[0] == Chess.MOVE:
            pass
        elif action[1] == Chess.LONGMOVE:
            pass
        elif action[2] == Chess.CASTLING:
            pass
        elif action[3] == Chess.PROMOTION_KNIGHT:
            pass
        elif action[4] == Chess.PROMOTION_QUEEN:
            pass
        elif action[5] == Chess.ENPASSANT:
            pass
        else:
            raise Exception("Unknown action taken: " + str(action))
        
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