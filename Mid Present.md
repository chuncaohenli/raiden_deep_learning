# Raiden AI

## Baseline Version
1. DQN
   1.  Wrap Environment
   
        Made serveral changes to Raiden game code to implement gym environment APIs.
    
      * Reconstructed the project and implement a RaidenENV class with step, render and reset methods.
    
      * Changed code structure to split the render and computation parts.
    
      * Wrote setup and initial scripts to register the raiden environment.
    
   2. Algorithm Design 
   
    * Reward Strategy Design
    
      If the player's fighter stay alive for each time step, get a 0.001 reward.<br />
      If it is crashed by enemies, get a negative reward of the hp of that enemy.<br />
      If it shoot one enemy down, get a reward of the hp of that enemy.<br />
      If it dies, get a 200 negative reward.<br />
    
    * Neural Network Design
    
      Made a convolution neural network:<br />
      Conv layers: (32, 8, 4), (64, 4, 2), (64, 3, 1)<br />
      Hidden layers: (256)
    
    * Hyperparameters Design
    
      gamma=1 (We set discount factor as 1 because future bonus is as important as current bonus)<br />
      max_timesteps=300000<br />
      exploration_fraction=0.6<br />
      exploration_final_eps=0.05 (This is a little bit higher than usual because this game is last quite long, we want the agent to explore enough)<br />
    
   3. Training Optimizing
   
    * Reduce nosise
    
      Remove background image and other irrelevant things
    
    * Simplify input
    
      Compreesed the size of the image we captured.<br />
      The orignal size is 700 * 900, after compressing it's 160 * 210 which reduced the computation in NN significantly.
    
    * FrameSkipping
    
      It's unnecssary to pass every image we captured to neural network every timestep. For each timestep, we repeat the same action for several times (random number), track the reward and coordinates of the fighter and only return the finnal result.
    
    * Split render and computation
    
      During training process, we don't render the screen and show the training process on the screen. Thus, we speed up the training process.

## Version 0

We implemented our own DQN algorithm to train the raiden game for about 40,000 iterations.

![Bad](resource/img_v0_good.gif)

The following advantages are observed:
- Plane knows how to avoid the enemies to some extent

The following issues are observed:
- Training is slow
- Plane keeps crashing the enemies after running for 5 seconds
- Plane gets stuck on the edge and does not have the motivation to shoot the enemies

The following improvements are made based on the observations:
- Adjust the hp of enemies so that they can be cleared after one shot
- Adjust the random actions probabilities so that the plane is more possible to explore around
- Speed up the game

## Version 1
We adjust the settings of the game and parameters of our algorithms. We train the new model for about 100,000 iterations.

![V1](resource/img_v0_good_2.gif)

The following advantages are observed:
- Plane shoots part of the enemies
- Game is running faster
- Plane has more freedom to explore

The following issues are observed:
- Plane still keeps crashing enemies 

The reason of crashing is possibly that the original reward function of DQN does not count the crashing immediatly, instead, it only reacts when one life is lost. Moreover, referenced by the reward function in Flappy bird, we shall give small reward for each time the plane survives. In such a case, reward will not stay the same number for a long time, which may result a difficulty in training.

The following improvements are made based on the observations:
- Adjust the reward function based on the previous content

## Version 2
We adjust the reward function and train the new model for about 100,000 iterations.

![V2](resource/img_vf_good_2.gif)

The following advantages are observed:
- Plane can almost clear all the Type I enemies without getting any harm

The following issues are observed:
- Plane cannot deal with the suicide type enemies

We find that the suicide type enemies are more difficult to deal with than other types. As a result, we adjust the order of the enemies so that they comes in the order from the easiest to the most difficult one.

![V2](resource/img_vf_good.gif)

![V2](resource/img_vf_tower.gif)

## Technical issue
1. Baseline integration


2. Cloud server video service
    When we train our model in cloud services, the following errors will come
    ```sh
    pygame.error: No available video device
    ```
    Set the video device to "dummy" will cause the loss function to be NAN, which     raises a problem in training. 
    We solved the method by setting VNC port in Google cloud and run our         algorithms via a VNC client server connected to the cloud service.
## Future Work

- Improve the DQN. Change the settings of the game so that the enemies will appear in a random initail point instead of a fixed point now. Moreover we could add noise on the movements of enemies to see if DQN could be stable and perform well.
- Implement A3C and PPO

## A3C implementation
We implemented A3C algorithm to train our agent for game Raiden. 
### Why A3C
DQN algorithm is very time-consuming. Using my own laptop to train the model for 10 hours will only finish about 40,000 iterations. So in order to find a more efficient algorithm, we find A3C, which is a improvement version of actor critic algorithm. Compared with other DQN algorithms,
it has several prons:
- Faster
- Simpler
- More robust
- Better scores
### A3C performance
