import chess
from lib.engine_wrapper import MinimalEngine
from .LearningEngineWrapper import LearningEngineWrapper

# Make LearningEngineWrapper accessible as an attribute of the homemade module
LearningEngineWrapper = LearningEngineWrapper

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def material_count(new_board, turn):
    """
    Count material in the new position for the player to move.
    """
    all_pieces = new_board.piece_map().values()
    material_difference = 0

    for piece in all_pieces:
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == turn:
            material_difference += value
        else:
            material_difference -= value
    return material_difference

def improved_score(new_board, turn):
    score = material_count(new_board, turn)
    
    # add extra score strategies
    if new_board.is_checkmate():
        score += 999999
    
    # Compute Space Controlled by current color
    space = 0
    for square in chess.SQUARES:
        if new_board.is_attacked_by(turn, square):
            space += 1
        if new_board.is_attacked_by(not turn, square):
            space -= 1
    score += space * (1/32)

    all_pieces = new_board.piece_map().items()

    for square, piece in all_pieces:
        if piece.color == turn:
            attacker_count = len(new_board.attackers(not turn, square))
            defender_count = len(new_board.attackers(turn, square))
            if attacker_count > defender_count:
                score -= PIECE_VALUES[piece.symbol().upper()]

    return score

class ScoreEngine(MinimalEngine):
    def __init__(self, *args, name=None):
        super().__init__(*args)
        self.name = name
        self.score_function = improved_score

    def move(self, board, my_time, my_inc, opp_time, opp_inc):
        """
        Given the board, return the best move according to the scoring function.
        """
        moves = list(board.legal_moves)

        best_move = None
        best_score = -float('inf')

        for move in moves:
            new_board = board.copy()
            new_board.push(move)

            score = self.score_function(new_board, board.turn)

            if score > best_score:
                best_move = move
                best_score = score

        return best_move


class DummyEngine(object):
    def __init__(self):
        pass    
        
    def play(self, board):
        moves = list(board.legal_moves)
        return random.choice(moves)