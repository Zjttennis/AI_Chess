import chess
import chess.engine
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm

piece_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

class Create_chess_board():
    def __init__(self):
        self.board = chess.Board()
    
    def is_game_over(self):
        return self.board.is_game_over()
        
    def reset(self):

        self.board.reset()
        return self.get_state()
    
    def get_state(self):

        state = np.zeros((6, 8, 8))
        chess_piece = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        for square, piece in self.board.piece_map().items():
            row, col = divmod(square, 8)
            if piece.color == chess.WHITE:
                state[chess_piece[piece.piece_type], row, col] = 1
            else:
                state[chess_piece[piece.piece_type], row, col] = -1
            # print(state)
        return state

    def eat_reward(self, move):

        captured_piece = self.board.piece_at(move.to_square) 
        self.board.push(move) 
        
        reward = 0
        if captured_piece:
            reward = reward + piece_value[captured_piece.piece_type]  


        if self.board.is_checkmate():
            reward = reward + 100  


        if self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = reward + 50 

        return reward, self.board.is_game_over()


    def step(self, move):
       
        reward, done = self.eat_reward(move) 
        state = self.get_state() 
        return state, reward, done

class Create_Deep_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*8*128, 512),
            nn.Linear(512, 4096)
        )
        
    def forward(self,input):
        return(self.layers(input))


def train_model(EPOCHS):
    for EPOCH in tqdm(range(EPOCHS)):
        board = Create_chess_board()
        
        while not board.board.is_game_over():
            state = board.get_state()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            state_tensor.requires_grad = True
            with torch.no_grad():
                move_scores = chess_ai(state_tensor)
                      
            move = select_best(move_scores, board)  
            next_state, reward, done = board.step(move) 
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            
            ##########################################################
            next_q_values = chess_ai(next_state_tensor)
            max_next_q = next_q_values.max(1)[0]
            target_q_value = reward + gamma * max_next_q

            ##########################################################
            current_q_value = move_scores[0][move.from_square * 64 + move.to_square]
            # target_q_value = target_q_value.unsqueeze(0)
            current_q_value = current_q_value.unsqueeze(0)
            loss = loss_function(current_q_value, target_q_value)
            
            # target_reward = torch.tensor([reward] * 4096, dtype=torch.float32).unsqueeze(0).to(device)
            # target_reward.requires_grad = True
            # loss = loss_function(move_scores, target_reward)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done: 
                break
    return chess_ai

def select_best(move_scores, board):
    legal_moves = list(board.board.legal_moves) 
    move_values = {move: move_scores[0][move.from_square * 64 + move.to_square].item() for move in legal_moves}
    best_move = max(move_values, key=move_values.get)  
    return best_move

def play_game():
    board = Create_chess_board()
    #print("欢迎进入国际象棋对战！")

    while not board.board.is_game_over():
        print(board.board)
        if board.board.turn == chess.WHITE:  # 玩家走白棋 player:white chess
            move = input("请输入你的走法（如 e2e4）：")
            try:
                move = chess.Move.from_uci(move)
                if move in board.board.legal_moves:
                    board.board.push(move)
                else:
                    print("无效的走法，请重新输入。")
                    continue
            except ValueError as e:
                print(f"输入格式错误：{e}。请输入正确的走法（如 e2e4）。")
                continue
        else:  # AI black chess
            print("AI 正在思考...")
            state = board.get_state()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                move_scores = chess_ai(state_tensor)
            
            move = select_best(move_scores, board)  # AI 选择最佳走法
            print(f"AI choice: {move}")
            board.board.push(move)

        if board.board.is_checkmate():
            print("checkmate, game over")
        elif board.board.is_stalemate() or board.board.is_insufficient_material():
            print("Forcing a Draw, game over")
        elif board.board.is_variant_draw():
            print("Draw, game over")

def save_model(ai, filename):
    torch.save(ai.state_dict(), filename)
    print(f"model save to: {filename}")



EPOCHS = 100000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

chess_ai = Create_Deep_NN()
chess_ai.to(device)
optimizer = optim.Adam(chess_ai.parameters(), lr=0.001)
loss_function = nn.MSELoss()
gamma = 0.99
file_name = "Chess.pt"

if __name__ == "__main__":
    print("Start training AI...")
    try:
        chess_ai.load_state_dict(torch.load(file_name))
        print("Model loaded successfully")
    except:
        print("Model loaded failed, starting to train a new model")
    chess_ai.train()  # swith to training mode
    model = train_model(EPOCHS)  
    # model.to(device)
    # print("Training completed, start the game！")
    # play_game()

    save_model(model, file_name)