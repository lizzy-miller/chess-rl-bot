import chess
from lib.engine_wrapper import MinimalEngine
from lib.learning_engine import LearningEngine  

class LearningEngineWrapper(MinimalEngine):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args)
        self.name = name

        # Load trained weights if specified
        weight_file = kwargs.get("weight_file", None)
        self.model = LearningEngine(name=name, weight_file=weight_file, epsilon=0.0)

    def search(self, board, time_limit, ponder, draw_offered, root_moves):
        """
        The lichess-bot expects `search` to return a chess.engine.PlayResult.
        """
        move = self.model.search(board)

        return chess.engine.PlayResult(move, None, info={"string": "LearningEngine move"})
