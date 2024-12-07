# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import numpy as np
import random
import torch
from Model import CNN
from train_model import *
import Chess
import time
import matplotlib.pyplot as plt

def load_model(fname, model):
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint['model'])
    return checkpoint['loss'], checkpoint['t_acc'], checkpoint['v_acc']

def accuracy_by_lastturn(model, boards, states, results, last_turn, extracted_by_game = 10, threshold = 0.5):
    indices = [i for i in range(boards.shape[0])][extracted_by_game-last_turn::extracted_by_game]
    model.eval()
    model = model.to(device_cpu)
    with torch.no_grad():
        correctly_predicted = 0
        for from_idx in range(0, len(indices), 40000):
            to_idx = min(from_idx + 40000, len(indices))
            predict = model(boards[indices[from_idx : to_idx]], states[indices[from_idx : to_idx]])
            correctly_predicted += torch.sum((predict >= threshold) == results[indices[from_idx : to_idx]]).item()
    print("Accuracy for last turn:", last_turn, correctly_predicted/len(indices))
    return correctly_predicted/len(indices)

def choose_next_move(model, game, depth, use_device = device_cpu):
    actions  = game.get_available_actions()
    if not actions:
        _, winner = game.has_ended()
        if winner == 'W':
            return ((1, 0))
        elif winner == 'B':
            return ((0, 0))
        else:
            return ((0.5, 0))
    if depth == 1:
        boards, states = [], []
        for action in actions:
            ngame = Chess.Chess(game)
            ngame.do_action(action)
            board, state = ngame.to_tensor()
            boards.append(board)
            states.append(state)
        boards, states = torch.stack(boards), torch.stack(states)
        with torch.no_grad():
            preds = model(boards.float().to(use_device), states.float().to(use_device))
        # White turn
        if game.turn == 0x40: 
            action_idx = torch.argmax(preds).item()
        else:
            action_idx = torch.argmin(preds).item()
        value = preds[action_idx].item()
        action = actions[action_idx]
        return (value, action)
    else:
        results = []
        for action in actions:
            ngame = Chess.Chess(game)
            ngame.do_action(action)
            value, _ = choose_next_move(model, ngame, depth-1, use_device)
            results.append((value, action))
        # White turn
        if game.turn == 0x40: 
            results.sort(key=lambda x: x[0], reverse=True) # Max first
        else:
            results.sort(key=lambda x: x[0], reverse=False) # Min first
        return results[0]

prng = random.Random()
prng.seed(SEED)

train, validation, test = load_games(prng)

LAST_GAME_TEST = False
PHASE_GAME_TEST = True
BAR_GAME_TEST = False
NEXT_MOVE_TEST = False

N = 10000

if LAST_GAME_TEST:
    validation_boards, validation_states, validation_results = load_cache(f'{PATH}validation', prng, validation, n_games=0, only_last = 10)
    validation_boards, validation_states, validation_results = validation_boards.float(), validation_states.float(), validation_results.reshape((-1,1)).float()
    
    test_boards, test_states, test_results = load_cache(f'{PATH}test', prng, test, n_games=0, only_last=10)
    test_boards, test_states, test_results = test_boards.float(), test_states.float(), test_results.float()
    
    predictions = []
    for act in ['relu', 'lrelu']:
        model = CNN(activation = act)
        loss, tacc, vacc = load_model(f'DiscoveryChess_model_{act}.pt', model)
        # Accuracy
        pred = []
        print("Validation accuracy")
        for i in range(1,11):
            p = accuracy_by_lastturn(model, validation_boards, validation_states, validation_results, last_turn=i, threshold=0.5)
            pred.append(p)
        predictions.append(pred)
        pred = []
        print("Test accuracy")
        for i in range(1,11):
            p = accuracy_by_lastturn(model, test_boards, test_states, test_results, last_turn=i, threshold=0.5)
            pred.append(p)
        predictions.append(pred)
    
    for i,(relu_val, relu_test, lrelu_val, lrelu_test) in enumerate(zip(predictions[0],predictions[1],predictions[2],predictions[3])):
        print(f'{i} & {relu_val*100:.2f}\\% & {relu_test*100:.2f}\\% & {lrelu_val*100:.2f}\\% & {lrelu_test*100:.2f}\\% \\\\')

if PHASE_GAME_TEST:
    results = {}
    for activation in ['relu', 'lrelu']:
        results[activation] = {'O': 0, 'M': 0, 'E': 0}
        model = CNN(activation = activation)
        loss, tacc, vacc = load_model(f'DiscoveryChess_model_{activation}.pt', model)
        model.eval()
        for i in range(N):
            moves = []
            while len(moves)<60:
                idx = int(prng.random() * len(test))
                moves = test[idx][1].split(',')
                result = int(test[idx][0] == 1)
            actidx = 0
            game = Chess.Chess()
            for phase in ['O', 'M', 'E']:
                if phase == 'O':
                    target = prng.randint(12, 29)
                elif phase == 'M':
                    target = prng.randint(30, 49)
                else:
                    target = prng.randint(50, min(70, len(moves)))
                while actidx < target:
                    game.do_action_algebraic(moves[actidx])
                    actidx += 1
                board, state = game.to_tensor()
                predict = model(torch.stack([board]), torch.stack([state]))
                if (predict >= 0.5).item() == result:
                    results[activation][phase] += 1
    for phase in ['O', 'M', 'E']:
        print(f'{phase} & {results["relu"][phase]*100/N:.2f}\\% & {results["lrelu"][phase]*100/N:.2f}\\% \\\\')

if BAR_GAME_TEST:
    results = {}
    correct = {}
    for activation in ['relu', 'lrelu']:
        results[activation] = [0 for i in range(70)]
        correct[activation] = [0 for i in range(70)]
        counts = [0 for i in range(70)]
        model = CNN(activation = activation)
        loss, tacc, vacc = load_model(f'DiscoveryChess_model_{activation}.pt', model)
        model.eval()
        for i in range(N):
            idx = int(prng.random() * len(test))
            moves = test[idx][1].split(',')
            result = int(test[idx][0] == 1)
            game = Chess.Chess()
            boards, states, targets = [],[],torch.Tensor([[result] for j in range(min(70, len(moves)))])
            for j in range(min(70, len(moves))):
                game.do_action_algebraic(moves[j])
                board, state = game.to_tensor()
                boards.append(board)
                states.append(state)
            with torch.no_grad():
                predictions = model(torch.stack(boards), torch.stack(states))
                difference = torch.abs( predictions - targets ).flatten()
            for j in range(min(70, len(moves))):
                results[activation][j] += difference[j].item()
                if(difference[j].item() < 0.5):
                    correct[activation][j] += 1
                counts[j] += 1
        for i in range(70):
            results[activation][i] /= counts[i]
    categories = [str(i+1) for i in range(70)]
    plt.figure(figsize=(16, 5))
    plt.xlim(-1, 70)
    plt.bar(categories, results['relu'], color='#F5921B', width = 0.4, label='ReLU')
    plt.bar(np.arange(len(categories)) + 0.44, results['lrelu'], color='#4394E5', width = 0.4, label='LeakyReLU')
    plt.xlabel('Number of actions')
    plt.ylabel('Difference from Target')
    plt.xticks([str(2*i+1) for i in range(70//2)])
    plt.legend()
    plt.show()
    
    
use_device = device

if NEXT_MOVE_TEST:
    durations = {}
    predictions = {}
    for activation in ['relu', 'lrelu']:
        durations[activation] = [0, 0 ,0]
        predictions[activation] = [0, 0 ,0]
        model = CNN(activation = activation)
        loss, tacc, vacc = load_model(f'DiscoveryChess_model_{activation}.pt', model)
        model.eval()
        model = model.to(use_device)
        for i in range(N):
            idx = int(prng.random() * len(test))
            moves = test[idx][1].split(',')
            result = int(test[idx][0] == 1)
            game = Chess.Chess()
            actidx = int(prng.random() * (len(moves)-1))
            for j in range(actidx):
                game.do_action_algebraic(moves[j])
            expected = Chess.Chess(game)
            expected.do_action_algebraic(moves[actidx])
            for depth in range(1,3+1):
                stime = time.time()
                _,action = choose_next_move(model, game, depth, use_device)
                ttime = time.time() - stime
                durations[activation][depth-1] += ttime
                pred = Chess.Chess(game)
                pred.do_action(action)
                predictions[activation][depth-1] += int(pred == expected)
        for depth in range(3):
            durations[activation][depth] /= N
            predictions[activation][depth] /= N
    
    print("Prediction accuracy:")
    for depth in range(3):
        print(f'{depth+1} & {predictions["relu"][depth]*100:.2f}\\% & {predictions["lrelu"][depth]*100:.2f}\\% \\\\')
    
    print("Time per prediction")
    for depth in range(3):
        print(f'{depth+1} & {durations["relu"][depth]}\\% & {durations["lrelu"][depth]}\\% \\\\')
    

