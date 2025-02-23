import torch
import chess
import chess.engine
import numpy as np
from chess_AI_model import Create_Deep_NN, Create_chess_board, select_best # import customized libraries 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
filename = "Chess.pt"
def load_model():
    model = Create_Deep_NN().to(device)
    model.load_state_dict(torch.load(filename))
    model.eval()
    print(f"Model loaded: {filename}")
    return model

# Run play_game
def play_game(model):
    board = Create_chess_board()
    #print("欢迎进入国际象棋对战！")

    while not board.board.is_game_over():
        print(board.board)
        if board.board.turn == chess.WHITE:  # player - white
            move = input("Please input your move (example: e2e4)")
            try:
                move = chess.Move.from_uci(move)
                if move in board.board.legal_moves:
                    board.board.push(move)
                else:
                    print("Invalid move, please re-input your move")
                    continue
            except ValueError as e:
                print(f"input format error  {e}  please check your move (如 e2e4)")
                continue
        else:  # AI - black
            print("AI is thinking ....")
            state = board.get_state()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                move_scores = model(state_tensor)

            move = select_best(move_scores, board)  # AI selection
            print(f"AI choice: {move}")
            board.board.push(move)

        if board.board.is_checkmate():
            print("checkmate, game over!")
            break
        elif board.board.is_stalemate() or board.board.is_insufficient_material():
            print("Draw, game over!")
            break
        elif board.board.is_variant_draw():
            print("Forced draw, game over!")
            break

if __name__ == "__main__":
    model = load_model("chess.pt")
    play_game(model)