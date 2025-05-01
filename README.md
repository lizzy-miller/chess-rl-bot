# Chess RL Bot

Welcome to my chess engine!

This repository contains a reinforcement learning (RL) agent trained to play chess via self-play. 

## About the Bot

The chess engine is built from scratch following Dr. Bert Huang's [youtube series](https://www.youtube.com/watch?v=rt50SJBZaOk&list=PLUenpfvlyoa0VSYPGou2kW9SSwI0xCgjI). It learns via Q-learning, using a linear value function. The bot plays as the white pieces as it trains against the black pieces. Learning occurs through temporal-difference updates with prioritized experience replay. It supports exploraiton via epsilon-greedy policies and softmax-baed move selection. 

## Credits
- Dr. Bert Huang's [YouTube series](https://www.youtube.com/watch?v=rt50SJBZaOk&list=PLUenpfvlyoa0VSYPGou2kW9SSwI0xCgjI)
- [lichess-bot github](https://github.com/lichess-bot-devs/lichess-bot) for allowing interactive play

--

## Learning Objective:
    * The engine needs to learn the relative values of pieces (e.g., pawns, knights, queens) purely though experience--meaning it is not told how much each piece is worth. 
    * The engine should also be learning the correlations between piece placement and outcomes: 
        * Even without explicit spatial knowledge, the model hsould gradually learn that having certain pieces on certain squares (e.g., a pawn in the middel) correlates with better rewards or winning outcomes. 

## Training Perspective: 
    * The engine learns from the White player's perspective only. 
    
-----

## 2. State Representation and Action Space

* The environment is a standard chessboard.

* The agent only receives partial information about the environment.

* The action space is a list of all available moves that the white engine can make. 

* Therefore, the agent is acting dynamically in a changing environment. 

----
The model does not have access to the full board layout directly. Instead, it interacts with the environment by evaluating the legal moves available at each step.

### 2.a How the Model Perceives the Environment

1. The legal moves available to the white engine at the given position
2. The material difference of the board. 
3. If white is checkmated.
----

--- 
### 2.b. What does the engine *not see*?

The model does not explicitly know how much each piece is worth.

- For example, it can evaluate that the material difference is +1 point, but it will not know (at least in the beginning) that capturing a pawn is worth 1 point.
- The model is expected to **learn the relative value of pieces** through experience.

In addition, the model does not explicitly know **which pieces are on which squares**.
- It only knows the available moves. (e.g., it knows that it can move a piece to the c3 square but does not explicitly know which piece is moving.)
- The model must **infer spatial and piece identity information** through training over time.

---

### 2.c. As a result, the model is not designed to explicitly consider strategies/heuristics like:


- Positional factors (e.g., "Is a rook on an open file?")
- King safety
- Center control
- Piece activity
- Tactical threats (e.g., forks, pins, skewers)

---

## 3. The Opponent

Also part of the environment is the opponent, which in this case is the **Black Engine**.

The White engine learns through **self-play**.

The opponent has a slight advantage in that:

- **The Black Engine has access to piece value knowledge — it selects moves based on a simple evaluation of material balance (e.g., pawns = 1, queens = 9).**

- This gives Black an early advantage during the initial learning stages.  
  - For example, the White engine does not initially realize that trading its queen for a pawn is a poor exchange.

In addition, the opponent's behavior is **not explicitly modeled** by the agent.  
- The White engine does not directly observe whether the opponent moved a pawn or a knight to a given square — it only sees the available legal moves and the resulting board state.

Instead, the opponent's moves are treated as part of the **environment’s natural transition dynamics**.

The Black engine does **not learn independently**.  
However, every **5000 games**, it updates its weights to match the White engine's current weights.  
This mechanism makes the opponent progressively stronger over time, encouraging the White engine to continue improving.

---

**Summary Takeaway**:  
The Black engine provides a consistent but evolving challenge for the White engine.  
The White engine must learn purely from the environment's responses without modeling the opponent explicitly.


---

This game of chess closely resembles the 'Fog of War' variant of chess: https://www.chess.com/variants/fog-of-war

---

## 4. Model/Algorithm

**Overview**: The aim of learning is for the model to learn the values of the chess pieces

* **Model Type:**
    * Simple linear model -- no neural networks
    * Weight Vector: 391-dimensions 
    

* $\epsilon$-greedy policy 
    * Exploration at a rate of 0.01 with a decay of 0.999 every 100 games played

* **Learning Algorithm:**
    * Q-learning with Temporal Difference (TD) error.

---

### 4.a Model Architecture


The model's architecture is delibertly simple to focus learning purely on reward signals rather than on a complex board encoding. 

* Each game had a limit of 300 max moves to prevent endless games. 
* A game stops during checkmate or draws according to standard chess, including:
    * Stalemate
    * 75 move rule
    * Insufficient Material
    * Fivefold repitition 

**Rewards:**
* (+ 100) for Winning
* (- 100) for Losing 
* (0) for Draws or Max moves reached.


--- 

#### 4.a.a What happens once a game is finished?

1. The model stores a record of all the board positions and moves it made during the game into a **Prioritized Replay Buffer**. (batchsize = 10)
2. Each experience is assigned a priority based on its immediate reward and updated over time based on the temporal difference (TD) error.
3. After accumulating enough experiences, the model samples a batch of important moves — favoring moves with higher priority — and applies **Q-learning updates**.
4. Specifically, for each sampled experience, the model uses the TD error between the predicted move value and the updated target to adjust its weights.
5. This process reinforces moves that led to favorable outcomes (material gain, checkmate) and penalizes moves that resulted in material loss or defeat.
6. Additionally, the final move of the game receives a large reward or penalty depending on whether the model won, lost, or drew the game.


--- 


## 5. Model Performance

* The model was trained over 120,000 total games
* The model trained for over 18 hours