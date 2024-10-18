from Chess import Chess
from model import CNN_3layers


game = Chess()
board_tensor, state_vector = game.to_tensor()

board_tensor = board_tensor.unsqueeze(0)  # Shape: [1, 12, 8, 8]
state_vector = state_vector.unsqueeze(0)  # Shape: [1, 22]

model = CNN_3layers(state_vector_size=22)

output = model(board_tensor, state_vector)

print(output)
