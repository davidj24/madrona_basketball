# File By File Description
**action.py:** The DiscreteActionDistributions class for sampling and getting the best of
<br>
<br>
**agent.py:** Agent class which is the neural network/model that is being trained. Also includes classes for normalizers particularly for observation and value normalization
<br>
<br>
**buffers.py:** The actual rolloutbuffer class
<br>
<br>
**controllers.py:** Controller classes that determine how agents are controlled during simulation: rule-based, human controlled, or by a policy
<br>
<br>
**env.py:** Class for the environment. Describes everything from input size and action buckets to stepping the worlds to the next timestep
<br>
<br>
**moving_avg.py:** 
<br>
<br>
**ppo_stats.py:**
<br>
<br>
## infer.py
Runs inference on model(s) for evaluation.

Key arguments: 
<br>
- --watch-model: Infers all versions of a model (across all iterations and generations as long as they're within the same directory) on a fixed set of episodes for evaluation of generations. 
<br>
- --checkpoint-0: The offensive agent checkpoint 
<br>
- --checkpoint-1: The defensive agent checpoint 
<br>
- --trainee-idx: 0 for offense, 1 for defense. so the program knows which one to use 
<br>
- --num_episodes: The number of episodes to infer
- --num-envs: The number of environments to run these episodes across.
<br>
<br>
## ppo.py
Runs training on a single agent using Proximal Policy Optimization
<br>
<br>
Key arguments;
<br>
<br>
- --seed: Just a run seed that makes training reproducable
- --torch-deterministic: Forces pytorch to use deterministic algorithms so training runs are deterministic based on seed
- --use-gpu
- --viewer: Displays one episode of training in world0 every log_every_n_iterations
- --full-viewer: Watch training in real time, but training goes 30x slower compared to normal

- --trainee-idx: 0 to train offense, 1 to train defense
- --trainee-checkpoint: The checkpoint to be used for whoever was labeled as the trainee
- --frozen-checkpoint: The checkpoint that is frozen and used as part of environment
