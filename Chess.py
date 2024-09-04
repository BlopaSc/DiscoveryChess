# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import numpy as np
from typing import *

class Chess:
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6
    WHITE = 8
    BLACK = 16
    
    def __init__(self, state = None):
        if state:
            self.state = state.state.copy()
            self.turn = state.turn
        else:
            self.state = np.zeros((8,8), dtype=np.int8)
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
            self.state[0,4] = Chess.KING | Chess.WHITE
            self.state[7,4] = Chess.KING | Chess.BLACK
            self.turn = 0
    
    def valid_position(self, position : tuple):
        return position[0] >= 0 and position[0] < 8 and position[1] >= 0 and position[1] < 8
    
    def add_valid_position(self, position : tuple, locations : set):
        if self.valid_position(position):
            locations.add(position)
    
    def get_attack_pawn(self, position : tuple, color : np.int8, locations : set):
        dy = 1 if color == Chess.WHITE else -1
        self.add_valid_position((position[0]+dy, position[1]+1), locations)
        self.add_valid_position((position[0]+dy, position[1]-1), locations)
        
    def get_attack_knight(self, position : tuple, color : np.int8, locations : set):
        self.add_valid_position((position[0]+2, position[1]+1), locations)
        self.add_valid_position((position[0]+2, position[1]-1), locations)
        self.add_valid_position((position[0]-2, position[1]+1), locations)
        self.add_valid_position((position[0]-2, position[1]-1), locations)
        self.add_valid_position((position[0]+1, position[1]+2), locations)
        self.add_valid_position((position[0]-1, position[1]+2), locations)
        self.add_valid_position((position[0]+1, position[1]-2), locations)
        self.add_valid_position((position[0]-1, position[1]-2), locations)
        
    def get_attack_bishop(self, position : tuple, color : np.int8, locations : set):
        for i in range(1, min(position)+1):
            npos = (position[0]-i, position[1]-i)
            if not self.state[npos]:
                self.locations.add(npos)
            elif not (self.state[npos] & color):
                self.locations.add(npos)
            else:
                break
        for i in range(1, min(position[0], 7-position[1])+1):
            npos = (position[0]-i, position[1]+i)
            if not self.state[npos]:
                self.locations.add(npos)
            elif not (self.state[npos] & color):
                self.locations.add(npos)
            else:
                break
        for i in range(1, min(7-position[0], position[1])+1):
            npos = (position[0]+i, position[1]-i)
            if not self.state[npos]:
                self.locations.add(npos)
            elif not (self.state[npos] & color):
                self.locations.add(npos)
            else:
                break
        for i in range(1, min(7-position[0], 7-position[1])+1):
            npos = (position[0]+i, position[1]+i)
            if not self.state[npos]:
                self.locations.add(npos)
            elif not (self.state[npos] & color):
                self.locations.add(npos)
            else:
                break
        
    
    def get_available_actions_pawn(self, position : tuple, color : np.int8):
        moves = []
        dy = (1 if color == Chess.WHITE else -1) if (position[0] and position[0]<7) else 0
        if not self.state[position[0]+dy, position[1]]:
            moves.append((position[0]+dy, position[1]))
            if (color == Chess.WHITE and position[0]==1 and not self.state[position[0]+2, position[1]]) or \
                (color == Chess.BLACK and position[0]==6 and not self.state[position[0]-2, position[1]]):
                moves.append(self.state[position[0]+(2*dy), position[1]])
        # TODO: Incorporate en passe
        # TODO: Incorporate attack
        # TODO: Incorporate promotion
            
    
    def get_available_moves(self):
        moves = []
        
    
    