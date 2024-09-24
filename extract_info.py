# Extracts information from the original .pgn files
# Stores them as a compressed .bin files which can later be loaded

# Conditional import, only required if parsing the original .pgn datasets
if __name__ == "__main__":
    import chess.pgn

import os
import time
import zlib

years = [' -1899', ' 1900-1949', ' 1950-1969', ' 1970-1989', ' 1990-1999', ' 2000-2004', ' 2005-2009', ' 2010-2014', ' 2015-2019', ' 2020', ' 2021', ' 2022', ' 2023']

def parse_original_sourcefiles():
    stime = time.time()
    for year in years:
        games_list = []
        output_path = f'./Dataset/dbLumbrasGigaBase{year}.bin'
        file_path = f'./Dataset/LumbrasGigaBase{year}.pgn'
        if not os.path.exists(file_path):
            print("Error: Could not find file", file_path)
            continue
        if os.path.exists(output_path):
            print("Skipping:", file_path, "output file already exists:", output_path)
            continue
        print("Parsing:", file_path)
        with open(file_path) as pgn:
            game_counter = 1
            while True:
                try:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
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
                    games_list.append(game_info['Winner'][0] + ','.join(game_info['Moves']))
                    game_counter += 1
                except:
                    print("utf-8 codec can't decode byte 0x80")
                    continue
            print(f"Read: {game_counter-1} games, Time taken: {time.time() - stime}")
        with open(output_path, 'wb') as out:
            print("Compressing at:", output_path)
            out.write(zlib.compress(bytes('\n'.join(games_list), encoding='utf-8'), level=9))
            print(f"Compressed successfully, Time taken: {time.time() - stime}")

def load_games():
    games = {}
    for year in years:
        games_list = []
        file_path = f'./Dataset/dbLumbrasGigaBase{year}.bin'
        if not os.path.exists(file_path):
            print("Failed to find file:", file_path)
            continue
        with open(file_path, 'rb') as binfile:
            read_file = zlib.decompress(binfile.read()).decode('utf-8')
            games_list = read_file.split('\n')
            games[year[1:]] = [({'W': 1, 'B': -1, 'D': 0}[g[0]], g[1:]) for g in games_list]
    return games

if __name__ == "__main__":
    print("Parsing files")
    parse_original_sourcefiles()
