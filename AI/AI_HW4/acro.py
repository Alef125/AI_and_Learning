import time
import gym
import random
import numpy as np

env = gym.make('Acrobot-v1')

num_states = 20000
max_episode_steps = 500


def discretizer(x, portion):
    if x < portion[0]: return 0
    for i in range(len(portion) - 1):
        if portion[i] <= x < portion[i + 1]: return i + 1
    if x >= portion[-1]: return len(portion)


def env_state_to_Q_state(stat):
    [cos_theta1, sin_theta1, cos_theta2, sin_theta2, thetaDot1, thetaDot2] = stat

    cos_theta1 = discretizer(cos_theta1, [-2 / 3, -1 / 3, 0, 1 / 3, 2 / 3])
    sin_theta1 = discretizer(sin_theta1, [-2 / 3, -1 / 3, 0, 1 / 3, 2 / 3])

    cos_theta2 = discretizer(cos_theta2, [-2 / 3, -1 / 3, 0, 1 / 3, 2 / 3])
    sin_theta2 = discretizer(sin_theta2, [-2 / 3, -1 / 3, 0, 1 / 3, 2 / 3])

    thetaDot1 = discretizer(thetaDot1, [0])
    thetaDot2 = discretizer(thetaDot2, [0])

    state = int(cos_theta1 + 6 * sin_theta1 + 36 * cos_theta2 + 216 * sin_theta2 + 1296 * thetaDot1 + 2592 * thetaDot2)
    return state


def Train():
    Qval = [[0] * 3 for i in range(num_states)]
    gama = 0.9
    alpha = 0.01

    while True:

        state = env_state_to_Q_state(env.reset())
        score = 0
        done = False
        step_count = 0

        while (not done) and step_count < max_episode_steps:
            # time.sleep(0.04)
            action = random.randint(0, 2)
            state, reward, done, _ = env.step(action)
            state = env_state_to_Q_state(state)
            step_count += 1
            score += int(reward)
            # env.render()

    Policy = [0] * num_states
    print("Policy : ", Policy)
    np.save('q_saved', Policy)


def Play():
    scores = []
    Policy = np.load('q_saved.npy')
    print("Policy : ", Policy)

    for episode_count in range(1000):
        episode_count += 1
        print('******Episode ', episode_count)
        state = env_state_to_Q_state(env.reset())

        score = 0
        done = False
        step_count = 0
        while not (done) and step_count < max_episode_steps:
            time.sleep(0.04)
            action = Policy[state]
            state, reward, done, _ = env.step(action)
            state = env_state_to_Q_state(state)
            step_count += 1
            score += int(reward)
            env.render()  # render current state of environment

        print('Score:', score)
        scores.append(score)

    print("Average score over 1000 run : ", np.array(scores).mean())

# Train()
#Play()
