import chess
import numpy as np
import sys
import codecs
import json
import pandas as pd
import time
import os
#from data.FEN.positions import fen_positions

#os.chdir("/sfs/gpfs/tardis/home/zrc3hc/CHESS_RL_RIV")


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


def material_count(new_board):
    """
    Count material in the new position for player to move.
    
    """
    
    all_pieces = new_board.piece_map().values()
    material_difference = 0
    
    for piece in all_pieces:
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == new_board.turn:
            material_difference += value
        else:
            material_difference -= value
    return material_difference

def checkmate_reward(prev_board, new_board):
    # If the move ended the game by checkmate
    if new_board.is_checkmate():
        # If it is still my turn after the move, I got checkmated (bad)
        if new_board.turn == prev_board.turn:
            return -100  # BAD: I got checkmated
        else:
            return 100   # GOOD: I checkmated the opponent
    elif new_board.is_stalemate():
        return 0
    elif new_board.is_insufficient_material():
        return 0
    elif new_board.is_seventyfive_moves():
        return 0
    elif new_board.is_fivefold_repetition():
        return 0
    else:
        return 0
    
def compute_reward(prev_board, prev_move, new_board):
    reward = material_count(new_board) - material_count(prev_board)
    reward += checkmate_reward(prev_board, new_board)
    return reward


            
class PrioritizedBuffer(list):
    def __init__(self, max_size):
        self.buffer = []
        self.priorities = []
        self.max_size = max_size
        self.cursor = 0
    
    def add_item(self, x, priority = 1.0): # x = experience
        if len(self.buffer) < self.max_size:
            self.buffer.append(x)
            self.priorities.append(priority)
        else:
            self.buffer[self.cursor] = x
            self.priorities[self.cursor] = priority
            self.cursor = (self.cursor + 1) % self.max_size
        
    def sample(self, batch_size, random_state):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = random_state.choice(len(self.buffer), size=batch_size, p=probs)
        sampled_experiences = [self.buffer[i] for i in indices]
        return sampled_experiences, indices
            

class LearningEngine():
    def __init__(self, *args, name = None, weights = None, weight_file = None, buffer_size = 10000, seed = 0, batch_size = 10, epsilon = 0.01):
        self.name = name
        
        self.random_state = np.random.RandomState(seed)
        #self.buffer = CircleBuffer(buffer_size) 
        self.buffer = PrioritizedBuffer(buffer_size)
        self.batch_size = batch_size
        self.epsilon = epsilon
        
        if weight_file:
            #load weights from file
            with codecs.open(weight_file, 'r', encoding='utf-8') as fopen:
                weight_test = fopen.read()
                weight_list = json.loads(weight_test)
            self.weights = np.array(weight_list)
        elif weights is None:
            #initialize weights
            starting_board = chess.Board()
            descriptor = self.features(starting_board)
            self.weights = np.random.rand(descriptor.size)
        else:
            self.weights = weights
        
    @staticmethod     
    def features(board):
        """
        Returns a numerical vector describing the board position.
        
        """
        
        all_pieces = board.piece_map().items()
        
        features = np.zeros(7)
        
        index = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        piece_grid = np.zeros((64, 6))
        
        
        
        for position, piece in all_pieces:
            value = -1 if piece.color == board.turn else 1
            
            type_index = index[piece.piece_type]
            
            features[index[piece.piece_type]] += value
    
            
            piece_grid[position, type_index] = value
            
        
        if board.is_checkmate():
            features[6] = 1
            
        return np.concatenate((features, piece_grid.ravel()))

    def action_score(self, board, move): # q-function
        """
        Returns the score of the given move.
        
        """
        
        
        # apply the move
        board.push(move)
        
        # get the features of the new position
        descriptor = self.features(board)

        board.pop()
        
        # get the score of the new position
        score = self.weights.dot(descriptor)
        
        return score

    def search(self, board, time_limit = None, ponder = None):
        
        moves = list(board.legal_moves)
        
        # Epsilon-greedy step
        
        if self.random_state.rand() < self.epsilon:
            # Exploration: Choose a Random Move
            sampled_move = self.random_state.choice(moves)
            
        else:
            # Exploitation: Choose the Best Move (SoftMax)
            
            scores = np.zeros(len(moves))
            
            for i, move in enumerate(moves):
                # apply the current candidate move
                scores[i] = self.action_score(board, move)
                
            probs = np.exp(scores - scores.max())
            
            probs /= np.sum(probs)
            
            samples = np.random.multinomial(1, probs)
            sampled_move = moves[np.min(np.argwhere(samples))]
        
        return sampled_move

    def save_weights(self, filepath):
        """
        Saves the weights to a file.
        
        """
        weight_list = self.weights.tolist()
        with codecs.open(filepath, 'w', encoding='utf-8') as fopen:
            json.dump(weight_list, fopen)
            
    def q_learn(self, reward, prev_board, prev_move, new_board, learning_rate = 0.01, discount = 1.0):
        # q(a, s) is estimate of discounted future reward after 
        #       making move a from s
                
                # q(a,s) <- q(a,s) + learning_rate * (reward + max_{a'} q(a',s') - q(a,s))
                # weights <- weights + learning_rate * 
                    #               (reward + max_future_score)
        
        # store position in buffer
        priority = abs(reward) + 1e-5 # Avoid zero priority
        
        self.buffer.add_item((reward, prev_board.copy(), prev_move, new_board), priority = priority) # for PrioritizedBuffer
        
        if len(self.buffer.buffer) < self.batch_size:
            # not enough samples to train
            return # Skip learning until the buffer has enough data. 
            
        # sample a batch with priority 
        sampled_batch, indices = self.buffer.sample(self.batch_size, self.random_state)
        
        # compute q-learning lookahead score
        
        for idx, (batch_reward, batch_prev_board, batch_prev_move, batch_new_board) in zip(indices, sampled_batch):

            moves = list(batch_new_board.legal_moves)
            
            if len(moves) == 0:
                max_future_score = 0
            else:
                max_future_score = max([self.action_score(batch_new_board, move) for move in moves])
                
            batch_prev_board.push(batch_prev_move)
            descriptor = self.features(batch_prev_board)
            batch_prev_board.pop()
            
            # td error = temporal difference error
                # Measures 'how surprosed' the agent is by the reward
                # td error = target - prediction
                    # target = reward + discount * max_future_score
                    # prediction = self.action_score(batch_prev_board, batch_prev_move)
                    
            prediction = self.action_score(batch_prev_board, batch_prev_move)
            target = batch_reward + discount * max_future_score
            td_error = target - prediction
            
            # update weights
            self.weights += learning_rate * td_error * descriptor
            
            
            # Update priority for this sample
            max_priority = 10.0
            self.buffer.priorities[idx] = min(abs(td_error) + 1e-5, max_priority)
            
            
        
        
        
if __name__ == "__main__":
    start_time = time.time()
    
    engine_white = LearningEngine(name="white", weight_file="mf2_weights/weights_4000.json", epsilon=0.095) # White Engine we're training
    board = chess.Board()
    engine_black = LearningEngine(name = "black", weights = None, weight_file = None, epsilon= 0.0) # Black Engine doesn't explore
    
    engine_black.weights[:7] = [1., 3., 3., 5., 9., 0., 25.] # Hardcoded weights for black
    #engine_white.weights = np.array([1., 1., 1., 1., 1., 0., 25.])
    
    wins = 0
    losses = 0
    draws = 0
    
    max_moves = 300
    max_games = 10000
    
    win_history = []
    loss_history = []
    draw_history = []
    
    epsilon_history = []
    # play games until interrupted
    try:
        
        for fen_idx, fen in enumerate(fen_positions):
            print(f"Training on FEN {fen_idx+1}/{len(fen_positions)}: {fen}")
            
            for game_idx in range(max_games):  
                if (wins + losses + draws) > 0 and (wins + losses + draws) % 5000 == 0:
                    print("Updating black to match learned weights")
                    engine_black.weights = engine_white.weights.copy()

                board = chess.Board(fen)  # <-- Start from this FEN
                white_positions = []
                white_moves = []

                while not board.outcome() and board.fullmove_number < max_moves:
                    if board.turn == chess.WHITE:
                        white_positions.append(board.copy())
                        move = engine_white.search(board, 1000, True)
                        white_moves.append(move)
                    else:
                        move = engine_black.search(board, 1000, True)

                    board.push(move)


                # After one game finishes
                outcome = board.outcome(claim_draw=True)

                learning_rate = 0.0001

                for i in range(len(white_moves) - 1):
                    reward = compute_reward(white_positions[i], white_moves[i], white_positions[i + 1])
                    engine_white.q_learn(reward, white_positions[i], white_moves[i], white_positions[i + 1], learning_rate=learning_rate)


                if outcome is None:
                    reward = 0
                    draws += 1
                    draw_history.append(1)
                    win_history.append(0)
                    loss_history.append(0)
                elif outcome.winner == chess.WHITE:
                    reward = 100
                    wins += 1
                    win_history.append(1)
                    loss_history.append(0)
                    draw_history.append(0)
                elif outcome.winner == chess.BLACK:
                    reward = -100
                    losses += 1
                    loss_history.append(1)
                    win_history.append(0)
                    draw_history.append(0)
                elif outcome.winner is None:
                    reward = 0
                    draws += 1
                    draw_history.append(1)
                    win_history.append(0)
                    loss_history.append(0)
                else:
                    raise ValueError(f"Unexpected outcome: {outcome}")

                last_board = white_positions[-1].copy()
                last_board.push(white_moves[-1])
                last_reward = compute_reward(white_positions[-1], white_moves[-1], last_board)
                engine_white.q_learn(last_reward, white_positions[-1], white_moves[-1], last_board)


                engine_white.weights = engine_white.weights.clip(min=-25, max=25)

                weights = engine_white.weights
                piece_labels = ["P", "N", "B", "R", "Q", "K", "M"]
                piece_output = ", ".join(f"{label}: {weight:.2f}" for label, weight in zip(piece_labels, weights))
                print(piece_output)

                if outcome is None:
                    print("Game finished: No official results (e.g., max moves reached)")
                else:
                    print(f"Game finished: {outcome.result()}")

                total_games = wins + losses + draws
                if total_games > 0:
                    print(f"Win rate: {wins / total_games:.2%}, Loss rate: {losses / total_games:.2%}, Draw rate: {draws / total_games:.2%}")

                if total_games % 100 == 0:

                    engine_white.save_weights(f"mf2_weights/weights_{total_games}.json")
                    print(f"Epsilon: {engine_white.epsilon:.4f}")

                    if engine_white.epsilon > 0.01:
                        engine_white.epsilon *= 0.9997

                    results_df = pd.DataFrame({
                        "Wins": win_history,
                        "Losses": loss_history,
                        "Draws": draw_history
                    })
                    results_df.to_csv("mf2_results/results.csv", index=False)
                    print("Results saved to mf2_results/results.csv")

                    epsilon_history.append(engine_white.epsilon)
                    epsilon_df = pd.DataFrame({"Epsilon": epsilon_history})
                    epsilon_df.to_csv("mf2_epsilon/epsilon.csv", index=False)
                    print("Epsilon history saved to mf2_epsilon/epsilon.csv")

                    elapsed_time = time.time() - start_time
                    hours, remainder = divmod(elapsed_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f"Elapsed time: {int(hours)}:{int(minutes):02}:{int(seconds):02}")
                    time_df = pd.DataFrame({
                        "Elapsed Time": [f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"]
                    })
                    time_df.to_csv("mf2_results/elapsed_time.csv", index=False)
            
            engine_white.save_weights(f"mf2_weights/weights_fen{fen_idx+1}.json")
            print(f"Saved weights after FEN {fen_idx+1}")




    except KeyboardInterrupt:
        print("Training interrupted. Saving results...")

        results_df = pd.DataFrame({
            "Wins": win_history,
            "Losses": loss_history,
            "Draws": draw_history
        })
        results_df.to_csv("mf2_results/results.csv", index=False)
        print("Results saved to mf2_results/results.csv")

        epsilon_df = pd.DataFrame({"Epsilon": epsilon_history})
        epsilon_df.to_csv("mf2_epsilon/epsilon.csv", index=False)
        print("Epsilon history saved to mf2_epsilon/epsilon.csv")