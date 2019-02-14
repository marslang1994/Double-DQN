import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)
class DDQN:
        def __init__(
            self,
            n_actions,
            n_dimension,
            learn_rate=0.005,
            reward_decay=0.9,
            epsilon_greedy_max=0.9,
            transfer_para_cyc=200,
            memory_size=3000,
            sample_size=32,
            epsilon_greedy_increment=None,
            output_tensorboard=False,
            double_dqn=True,
            sess=None
    ):  
            self.double_dqn = double_dqn
            self.n_actions = n_actions
            self.n_dimension = n_dimension
            self.lr = learn_rate
            self.gamma = reward_decay
            self.epsilon_max = epsilon_greedy_max
            self.transfer_para_cyc = transfer_para_cyc
            self.memory_size = memory_size
            self.sample_size = sample_size
            self.epsilon_increment = epsilon_greedy_increment
            self.epsilon = self.epsilon_max if epsilon_greedy_increment is None else 0

            self.steps_counter = 0
            self.memory = np.zeros((self.memory_size, n_dimension*2+2)) 
            self._build_network()

            tar_params = tf.get_collection('tar_net_para')
            eva_params = tf.get_collection('eva_net_para')
            self.transfer_para_opt = [tf.assign(t, e) for t, e in zip(tar_params, eva_params)]

            self.cost_trajectory = []

            if sess is None:
                self.sess = tf.Session()
                self.sess.run(tf.global_variables_initializer())
            else:
                self.sess = sess
                
            if output_tensorboard:
                tf.summary.FileWriter("logs/",self.sess.graph)
        
        def _build_network(self):
            self.s = tf.placeholder(tf.float32, [None, self.n_dimension],name='s')
            self.q_target = tf.placeholder(tf.float32, [None, self.n_actions],name='q_target') 

            with tf.variable_scope('evaluation_network'):
                para = ['eva_net_para', tf.GraphKeys.GLOBAL_VARIABLES]
                n1 = 20
                w_0 = tf.random_normal_initializer(0., 0.3)
                b_0 = tf.constant_initializer(0.1)
                with tf.variable_scope('layer_1'):
                    w_1 = tf.get_variable('w_1', [self.n_dimension, n1], initializer=w_0, collections=para)
                    b_1 = tf.get_variable('b_1', [1, n1], initializer=b_0, collections=para)
                    output_1 = tf.nn.relu(tf.matmul(self.s, w_1) + b_1)
                with tf.variable_scope('layer_2'):
                    w_2 = tf.get_variable('w_2', [n1, self.n_actions], initializer=w_0, collections=para)
                    b_2 = tf.get_variable('b_2', [1, self.n_actions], initializer=b_0, collections=para)
                    self.q_evaluation = tf.matmul(output_1, w_2) + b_2

            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_evaluation))
            with tf.variable_scope('train'):
                self._train_opt = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)        
            self.s_next = tf.placeholder(tf.float32, [None, self.n_dimension], name='s_next') 
            with tf.variable_scope('target_network'):
                para = ['tar_net_para', tf.GraphKeys.GLOBAL_VARIABLES]
                with tf.variable_scope('layer_1'):
                    w_1 = tf.get_variable('w_1', [self.n_dimension, n1], initializer=w_0, collections=para)
                    b_1 = tf.get_variable('b_1', [1, n1], initializer=b_0, collections=para)
                    output_1 = tf.nn.relu(tf.matmul(self.s_next, w_1) + b_1)
                with tf.variable_scope('layer_2'):
                    w_2 = tf.get_variable('w_2', [n1, self.n_actions], initializer=w_0, collections=para)
                    b_2 = tf.get_variable('b_2', [1, self.n_actions], initializer=b_0, collections=para)
                    self.q_eva_next = tf.matmul(output_1, w_2) + b_2

        def replay_storage(self, s, a, r, s_next):
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0        
            record = np.hstack((s, [a, r], s_next))
            index_memory = self.memory_counter % self.memory_size
            self.memory[index_memory, :] = record
            self.memory_counter += 1

        def select_action(self, observation):
            observation = observation[np.newaxis, :]
            actions_value = self.sess.run(self.q_evaluation, feed_dict={self.s: observation})

            if not hasattr(self, 'q_trajectory'):  # 记录选的 Qmax 值
                self.q_trajectory = []
                self.q_current = 0
            #self.q_current = self.q_current*0.7 + 0.3 * np.max(actions_value)
            self.q_current = np.max(actions_value)
            self.q_trajectory.append(self.q_current)

            if np.random.uniform() < self.epsilon:
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions) 
            return action

        def train(self):
            if self.steps_counter%self.transfer_para_cyc == 0:
                self.sess.run(self.transfer_para_opt)
                print('parameters transfered')

            if self.memory_counter <= self.memory_size:
                train_index = np.random.choice(self.memory_counter, size=self.sample_size)
            else:
                train_index = np.random.choice(self.memory_size, size=self.sample_size)
            sample_train = self.memory[train_index, :]

            q_evaluation = self.sess.run(self.q_evaluation, {self.s: sample_train[:, :self.n_dimension]})
            q_eva_next, q_find_index = self.sess.run([self.q_eva_next, self.q_evaluation], 
                                                     feed_dict = {self.s_next: sample_train[:, -self.n_dimension:],
                                                                 self.s: sample_train[:, -self.n_dimension:]})

            sample_index = np.arange(self.sample_size, dtype=np.int32)
            act_optimal_index = sample_train[:, self.n_dimension].astype(int)

            if self.double_dqn:
                action_choose_index = np.argmax(q_find_index, axis=1) 
                q_nex_chosen = q_eva_next[sample_index, action_choose_index]
            else:
                q_nex_chosen = np.max(q_eva_next, axis=1) 

            reward = sample_train[:, self.n_dimension + 1]

            q_target = q_evaluation.copy()
            q_target[sample_index, act_optimal_index] = reward + self.gamma * q_nex_chosen

            _, self.cost = self.sess.run([self._train_opt, self.loss],
                                         feed_dict={self.s: sample_train[:, :self.n_dimension],
                                                    self.q_target: q_target})

            self.cost_trajectory.append(self.cost)

            if self.epsilon < self.epsilon_max:
                self.epsilon += self.epsilon_increment
            self.steps_counter += 1