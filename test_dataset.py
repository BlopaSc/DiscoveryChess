# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import extract_info
import Chess
import time
import torch

games = extract_info.load_games(as_single_array = True)
chess = Chess.Chess()

skip = {}

t0 = time.time()

stack = []

for g,game in enumerate(games):
    if g in skip: continue
    chess.reset()
    if g % 100 == 0:
        print("Testing game:", g)
        if g and g % 1000 == 0: print("Time taken:", time.time()-t0, g/(time.time()-t0), "g/sec")
    try:
        for action in game[1].split(','):
            chess.do_action_algebraic(action, check=False)
            stack.append(chess.to_tensor())
    except Exception as e:
        print(e)
        raise Exception("Error while executing game: " + game[1] + "\nAs pgn: " + extract_info.to_pgn(game[1]))
        
    if g == 1000:
        break
    
maps = torch.stack(tuple(x[0] for x in stack))
states = torch.stack(tuple(x[1] for x in stack))

print("Time taken:", time.time() - t0)
