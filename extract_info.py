import chess.pgn
import json
import os

from os import listdir
from os.path import isfile, join

path = './Dataset'
# files = [f for f in listdir(path) if isfile(join(path, f))]
# years = [' -1899', ' 1900-1949', ' 1950-1969', ' 1970-1989', ' 1990-1999', ' 2000-2004', ' 2005-2009', ' 2010-2014', ' 2015-2019', ' 2020', ' 2021', ' 2022', ' 2023']
years = [' 2000-2004', ' 2005-2009', ' 2010-2014', ' 2015-2019', ' 2020', ' 2021', ' 2022', ' 2023']

# print(files)

# for file in files:
for year in years:
    games_list = []

    output_dir = './Clean_Dataset/'
    os.makedirs(output_dir, exist_ok=True)
    # file_path = join(path, file)
    file_path = f'./Dataset/LumbrasGigaBase{year}.pgn'
    
    with open(file_path) as pgn:
        game_counter = 1
        while True:
            try:
                game = chess.pgn.read_game(pgn)
                
                if game is None:
                    break
                print(game.headers)
                game_info = {}
                game_info['Moves'] = []

                if game.headers['Result'] == '1-0':
                    game_info['Winner'] = 'White'
                elif game.headers['Result'] == '0-1':
                    game_info['Winner'] = 'Black'
                else:
                    game_info['Winner'] = 'Draw'

                board = game.board()

                for move in game.mainline_moves():
                    game_info['Moves'].append(board.san(move))  
                    board.push(move)  

                games_list.append(game_info)
                
                # print(f"Processed game {game_counter}")
                game_counter += 1
            except:
                print("utf-8 codec can't decode byte 0x80")
                continue
    
    output_file_path = os.path.join(output_dir, f'LumbrasGigaBase{year}.json')
    
    with open(output_file_path, "w") as output_file:
        json.dump(games_list, output_file, indent=4)
    print(f'Finished writing {output_file_path}')

