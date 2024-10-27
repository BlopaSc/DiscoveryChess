# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

# Project imports
from Chess import Chess
from Model import CNN
import extract_info

# Libraries
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Randomness seed
SEED = 777

# Validation and test dataset splits
VAL_TEST_SET = 0.1
PRECOMP = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loads the games and splits them into train, validation and test datasets
def load_games(prng):
    games = extract_info.load_games(as_single_array = True, exclude_draws = True, checkmates_only = True)
    N = len(games)
    # Split games
    prng.shuffle(games)
    val_part = int( N*(1 - (2*VAL_TEST_SET)) )
    test_part = int( N*(1 - VAL_TEST_SET) )
    train, validation, test = games[:val_part], games[val_part:test_part], games[test_part:]
    return train, validation, test

# Returns the game tensors and target output from a dataset:
    # prng: Random object to use for sampling
    # dataset: Dataset from which to sample games
    # n_games: Number of games to take as sample, if 0 takes all
    # skip_first: If not None, then the first skip_first turns will be ommited from the state extraction
    # only_last: If not None, then only the last only_last turns will be considered for the state extraction
    # k_states: If not None, then only k_states will be extracted from the game, chosen at random from the available states
def game_tensors(prng, dataset, n_games, skip_first = None, only_last = None, k_states = None):
    if n_games and n_games != len(dataset):
        game_indices = prng.sample([i for i in range(len(dataset))], n_games)
    else:
        game_indices = [i for i in range(len(dataset))]
    chess = Chess()
    boards = []
    states = []
    results = []
    for game_index in game_indices:
        chess.reset()
        winner,actions = dataset[game_index]
        actions = actions.split(',')
        extract_from = skip_first if skip_first else 0
        if only_last: extract_from = max(extract_from, len(actions) - only_last)
        if k_states:
            # TODO: Implement if desired, random sample from states
            pass
        else:
            for a,action in enumerate(actions):
                try:
                    chess.do_action_algebraic(action, check=False)
                except Exception as e:
                    print(e)
                    raise Exception("Error while executing game: " + dataset[game_index][1] + "\nAs pgn: " + extract_info.to_pgn(dataset[game_index][1]))
                if a >= extract_from:
                    board, state = chess.to_tensor()
                    boards.append(board)
                    states.append(state)
                    results.append(1 if winner==1 else 0)
    return torch.stack(boards), torch.stack(states), torch.Tensor(results).reshape((-1, 1))

# Allows to load any precalculate cache, used for the last 10 turns of the test and validation datasets
def load_cache(cache_name, prng, dataset, n_games, **kwargs):
    name = cache_name + '.cache'
    if os.path.exists(name):
        data = torch.load(name)
        boards, states, results = data['boards'].to(torch.float32), data['states'].to(torch.float32), data['results'].to(torch.float32)
    else:
        print("Building cache:", cache_name)
        boards, states, results = game_tensors(prng, dataset, n_games, **kwargs)
        torch.save({
            'boards': boards.to(torch.uint8),
            'states': states.to(torch.uint8),
            'results': results.to(torch.uint8)
        }, name)
    return boards, states, results

def train_model():
    EPOCHS = 100
    NGAMES = 1000
    CACHES = 50
    MINIBATCHES = 500
    torch.manual_seed(SEED)
    model = CNN()
    prng = random.Random()
    # Initiliaze model
    model.train()
    model.zero_grad()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Prepare datasets
    prng.seed(SEED)
    # Load games
    train, validation, test = load_games(prng)
    # Prepare precomputed tensor caches (about 55 Gb, about 6-7 hours de compute)
    N_per_cache = []
    for c in range(CACHES):
        from_idx = (c*len(train))//CACHES
        to_idx = ((c+1)*len(train))//CACHES
        _,_,results = load_cache(f'train{c}', prng, train[from_idx:to_idx], n_games=0, skip_first=5)
        N_per_cache.append(results.shape[1])
    N = sum(N_per_cache)
    N_per_cache = np.cumsum(N_per_cache)
    # Prepare validation cache (about 0.75 Gb, about 8 minutes to compute)
    validation_boards, validation_states, validation_results = load_cache('validation', prng, validation, n_games=0, only_last = 10)
    # Prepare everything else
    loss_fn = nn.BCELoss()
    i = 0
    while i < EPOCHS:
        # Minibatch training
        prng.seed(SEED*(i+1))
        indices = prng.shuffle([i for i in range(N)])
        loss_epoch = 0
        predicted_correctly = 0
        for m in range(MINIBATCHES):
            from_idx = (m*N)//MINIBATCHES
            to_idx = ((m+1)*N)//MINIBATCHES
            minibatch_indices = indices[from_idx : to_idx]
            # Load minibatch from files
            minibatch_indices.sort()
            mini_idx = 0
            minibatch_boards = []
            minibatch_states = []
            minibatch_results = []
            for c in range(CACHES):
                limit = mini_idx
                while limit < len(minibatch_indices) and minibatch_indices[limit] < N_per_cache[c]:
                    limit += 1
                if limit == mini_idx: continue
                boards, states, results = load_cache(f'train{c}', prng, [], n_games=0, skip_first=5)
                minibatch_boards.append(boards[minibatch_indices[mini_idx : limit]])
                minibatch_states.append(states[minibatch_indices[mini_idx : limit]])
                minibatch_results.append(results[minibatch_indices[mini_idx : limit]])
                mini_idx = limit
            boards = torch.cat(minibatch_boards, dim=0)
            states = torch.cat(minibatch_states, dim=0)
            results = torch.cat(minibatch_results, dim=0)
            # Train
            model.zero_grad()
            boards, states, results = boards.to(device=device), states.to(device=device), results.to(device=device)
            predict = model(boards, states)
            loss = loss_fn(predict, results)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            predicted_correctly += torch.sum((predict >= 0.5) == results).item()
        boards, states, results = None, None, None
        # TODO: Calculate validation?
        print(f"Loss at epoch {i}: {loss_epoch}, TA: {predicted_correctly/N}")
        i += 1
    return model

if __name__ == "__main__":
    # Train model
    train_model()




