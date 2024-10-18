from Chess import Chess
from model import CNN


game = Chess()
board_tensor, state_vector = game.to_tensor()

board_tensor = board_tensor.unsqueeze(0)  # Shape: [1, 12, 8, 8]
state_vector = state_vector.unsqueeze(0)  # Shape: [1, 22]

model = CNN(input_channels=12, output_channels=[32, 64, 128], kernel_size=3, state_vector_size=22, num_layers=3)

output = model(board_tensor, state_vector)

print(output)
