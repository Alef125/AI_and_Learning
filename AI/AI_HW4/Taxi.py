# random agent on Taxi-v2
import gym

env = gym.make("Taxi-v2")

P = env.env.P
# P[s][a] contains array of tuples.
# each of them is in form of (p, next_s, r, done)
# where "p" is the transition probability of
# going from "s" to "next_s" with action "a" getting "r" reward;
# "done" denotes whether the episode is finished or not.

observation = env.reset()
score = 0
# since random agent might not be able to finish the game, I used for with 1000 steps instead of while True
for _ in range(1000):
    # env.reset() gives the initial position of taxi and state is a number between 0 and 499
    state = env.reset()
    # env.render() renders the map with current state
    env.render()
    # action is a number between 0 and 5. your task is to replace this random action with your own policy
    action = env.action_space.sample()  # your agent here (this takes random actions)
    # env.step() takes the game to the next state and returns next_state, reward, done, info.
    state, reward, done, info = env.step(action)

    score += reward
    if done:
        state = env.reset()
        break
print("score: ", score)
env.close()
