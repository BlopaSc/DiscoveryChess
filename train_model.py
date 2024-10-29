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
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Fast access read-storage, recommended SSD (will only write once when creating caches, will read several Tbs of data)
PATH = 'C:/Data/'

# Randomness seed
SEED = 777

# Validation and test dataset splits
VAL_TEST_SET = 0.1
PRECOMP = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

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
        boards, states, results = data['boards'], data['states'], data['results']
    else:
        print("Building cache:", cache_name)
        boards, states, results = game_tensors(prng, dataset, n_games, **kwargs)
        boards, states, results = boards.to(torch.uint8), states.to(torch.uint8), results.to(torch.uint8)
        torch.save({
            'boards': boards,
            'states': states,
            'results': results
        }, name)
    return boards, states, results

def train_model():
    # Estimated data usage per sample: (12*8*8 + 22 + 1)*4 = 3164 bytes/sample
    EPOCHS = 20
    CACHES = 50 # Splits the training data into CACHES files which will be read from storage
    BATCH_LOAD_SIZE = 10000000 # Or aprox 8 Gbs if keeping them as uint8, divisible by BATCH_SIZE
    BATCH_SIZE = 40000
    torch.manual_seed(SEED)
    model = CNN(activation = 'lrelu')
    prng = random.Random()
    # Initiliaze model
    model.train()
    model.zero_grad()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Prepare datasets
    prng.seed(SEED)
    # Check if missing caches
    missing = (not os.path.exists(f'{PATH}validation.cache')) or not os.path.exists(f'{PATH}sizes.cache') or any((not os.path.exists(f'{PATH}train{c}.cache')) for c in range(CACHES))
    if missing:
        # Load games
        print("Loading games")
        train, validation, test = load_games(prng)
        # Prepare precomputed tensor caches (about 55 Gb, about 6-7 hours de compute)
        N_per_cache = []
        print("Pre-computing caches")
        for c in range(CACHES):
            from_idx = (c*len(train))//CACHES
            to_idx = ((c+1)*len(train))//CACHES
            _,_,results = load_cache(f'{PATH}train{c}', prng, train[from_idx:to_idx], n_games=0, skip_first=5)
            N_per_cache.append(results.shape[0])
            results = []
        N = sum(N_per_cache)
        N_per_cache = [0] + list(np.cumsum(N_per_cache))
        torch.save({
            'N': N,
            'N_per_cache': np.array(N_per_cache),
        }, f'{PATH}sizes.cache')
    else:
        train, validation, test = [],[],[]
        data = torch.load(f'{PATH}sizes.cache')
        N = data['N']
        N_per_cache = list(data['N_per_cache'])
    print("N-sample:", N)
    print("N_per_cache:", N_per_cache)
    # Prepare validation cache (about 0.75 Gb, about 8 minutes to compute)
    validation_boards, validation_states, validation_results = load_cache(f'{PATH}validation', prng, validation, n_games=0, only_last = 10)
    # Have them ready: about 4 Gb to keep in memory
    validation_boards, validation_states, validation_results = validation_boards.float(), validation_states.float(), validation_results.reshape((-1,1)).float()
    # Prepare everything else
    loss_fn = nn.BCELoss()
    i = 0
    losses = []
    t_accuracies = []
    v_accuracies = []
    print("Starting training")
    while i < EPOCHS:
        # Minibatch training
        prng.seed(SEED*(i+1))
        indices = [i for i in range(N)]
        prng.shuffle(indices)
        loss_epoch = 0
        val_predicted_correctly = 0
        predicted_correctly = 0
        training_samples = 0
        stime = time.time()
        for m in range(N//BATCH_LOAD_SIZE):
            from_idx = m * BATCH_LOAD_SIZE
            to_idx = (m+1) * BATCH_LOAD_SIZE
            minibatch_indices = indices[from_idx : to_idx]
            # Load minibatch from files
            minibatch_indices.sort()
            mini_idx = 0
            minibatch_boards = []
            minibatch_states = []
            minibatch_results = []
            for c in range(CACHES):
                limit = mini_idx
                while limit < len(minibatch_indices) and minibatch_indices[limit] < N_per_cache[c+1]:
                    limit += 1
                if limit == mini_idx: continue
                boards, states, results = load_cache(f'{PATH}train{c}', prng, [], n_games=0, skip_first=5)
                adjusted_indices = [i - N_per_cache[c] for i in  minibatch_indices[mini_idx : limit]]
                minibatch_boards.append(boards[adjusted_indices])
                minibatch_states.append(states[adjusted_indices])
                minibatch_results.append(results[adjusted_indices])
                mini_idx = limit
                # Deallocate
                boards, states, results = None, None, None
            boards = torch.cat(minibatch_boards, dim=0)
            states = torch.cat(minibatch_states, dim=0)
            results = torch.cat(minibatch_results, dim=0)
            # Deallocate
            minibatch_boards, minibatch_states, minibatch_results = None, None, None
            for b in range(BATCH_LOAD_SIZE//BATCH_SIZE):
                from_idx = b * BATCH_SIZE
                to_idx = (b+1) * BATCH_SIZE
                # Train
                model.zero_grad()
                mboards = boards[from_idx : to_idx].float()
                mstates = states[from_idx : to_idx].float()
                mresults = results[from_idx : to_idx].float()
                mboards, mstates, mresults = mboards.to(device=device), mstates.to(device=device), mresults.to(device=device)
                predict = model(mboards, mstates)
                loss = loss_fn(predict, mresults)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                predicted_correctly += torch.sum((predict >= 0.5) == mresults).item()
                mboards, mstates, mresults = None, None, None
            training_samples += BATCH_LOAD_SIZE
        with torch.no_grad():
            val_indices = [i for i in range(validation_results.shape[0])]
            model = model.to(device_cpu)
            val_predicted_correctly = 0
            model.eval()
            for from_idx in range(0, validation_results.shape[0], BATCH_SIZE):
                to_idx = min(from_idx + BATCH_SIZE, validation_results.shape[0])
                predict = model(validation_boards[val_indices[from_idx : to_idx]], validation_states[val_indices[from_idx : to_idx]])
                val_predicted_correctly += torch.sum((predict >= 0.5) == validation_results[val_indices[from_idx : to_idx]]).item()
            model = model.to(device)
        model.train()
        losses.append(loss_epoch)
        t_accuracies.append(predicted_correctly/training_samples)
        v_accuracies.append(val_predicted_correctly/len(val_indices))
        print(f"Loss at epoch {i}: {loss_epoch}, TA: {predicted_correctly/training_samples}, VA: {val_predicted_correctly/len(val_indices)} , Time-epoch: {time.time() - stime}")
        i += 1
    torch.save({
        'epoch': i,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'loss': losses,
        't_acc': t_accuracies,
        'v_acc': v_accuracies
    }, 'DiscoveryChess_model_lrelu.pt')
    return model

if __name__ == "__main__":
    # Train model
    train_model()




