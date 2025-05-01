import random
import chess

class DummyEngine(object):
    def __init__(self):
        pass    
        
    def play(self, board):
        moves = list(board.legal_moves)
        return random.choice(moves)