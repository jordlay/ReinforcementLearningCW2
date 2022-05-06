# ReinforcementLearningCW2
Both jupyter notebook and html versions of each file are available and are identical.

# Master 1 of 4
Master 1 of 4 contains the results for an agent who behaves randomly in the Pendulum-v1, BipedalWalker-v3, and BipedalWalkerHardcore-v3 which form baselines for us to compare future results. It also contains our DDPG implementation for Pendulum-v1 and BipedalWalker-v3. 

# Master 2 of 4
Master 2 of 4 contains our implementation of TD3 for BipedalWalker-v3 with a hidden size of 128 and a batch size of 128.

# Master 3 of 4
Master 3 of 4 contains our implementation of TD3 for BipedalWalker-v3 with a hidden size of 64 and a batch size of 64. This agent is then tested on BipedalWalkerHardcore-v3. The agent is then trained on BipedalWalkerHardcore-v3 with hidden size of 64 and a batch size of 64 to ensure a reasonable runtime.

# Master 4 of 4
Master 4 of 4 contains our implementation of TD3 for BipedalWalkerHardcore-v3 with hidden size of 64 and a batch size of 64 to ensure a reasonable runtime. This allows us to compare the performance of an agent who learns to adapt from BipedalWalker-v3 to BipedalWalkerHardcore-v3 (Master 3 of 3) against an agent who learns BipedalWalkerHardcore-v3 from scratch (Master 4 of 4).
