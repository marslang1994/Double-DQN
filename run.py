import gym
from Double_DQN import DDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()

env = gym.make('Pendulum-v0')
env.unwrapped
env.seed(1)
memo_size = 3000
action_space = 11


with tf.variable_scope('DQN'):
    DQN = DDQN(
        n_actions=action_space, n_dimension=3, memory_size=memo_size,
        epsilon_greedy_increment=0.001, double_dqn=False, sess=sess
    )
    
with tf.variable_scope('DDQN'):
    DDQN = DDQN(
        n_actions=action_space, n_dimension=3, memory_size=memo_size,
        epsilon_greedy_increment=0.001, double_dqn=True, sess=sess, output_tensorboard=True)
    
def Run(NN):
    
    observation = env.reset()
    num_steps = 0
    
    while True:

        action = NN.select_action(observation)

        f_action = (action-(action_space-1)/2)/((action_space-1)/4)   # convert to [-2 ~ 2] float actions
        observation_next, reward, done, info = env.step(np.array([f_action]))

        reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

        NN.replay_storage(observation, action, reward, observation_next)

        if num_steps > memo_size:   # learning
            NN.train()

        if num_steps - memo_size > 20000:   # stop game
            break

        observation = observation_next
        num_steps += 1
    return NN.q_trajectory

q_DQN = Run(DQN)
q_DDQN = Run(DDQN)

plt.plot(np.array(q_DQN), c='r', label='DQN')
plt.plot(np.array(q_DDQN), c='b', label='DDQN')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()




