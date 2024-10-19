# RL-NLP_agent-text_based_games
RL agent for Text-based Game. The text-based used for this purpose was Zork I. The game runs on a command line interface that displays description of the place the player is currently in, using this description the RL/NLP model is made to select a command out of the action space to execute it.


# A2C - Advantage Actor Critic
The [A2C paper](https://arxiv.org/abs/1602.01783v2) explains the inclusion of Advantage to the Actor-Critic algorithm. The proposed model uses the A2C algorithm but using DistilBERT and attention mechanism to weight the previous outputs. The technique helps the model to retain knowledge.



