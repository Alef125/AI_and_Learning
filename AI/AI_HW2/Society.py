import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
matplotlib.use("TkAgg")


class Society:

    def __init__(self, law_book=None, initial_state=None, size=(15, 15), p=0.5):

        if initial_state is None:
            self.size = size
            self.state = np.random.binomial(1, p, self.size).astype(int)
        else:
            self.size = initial_state.shape()
            self.state = initial_state

        if law_book is None:
            self.law_book = {}
            for i in range(256):
                self.law_book[Society.base_convert(i, 2, 8)] = np.random.randint(0, 2)

        else:
            self.law_book = law_book

    def reset_state(self, p=0.5):
        self.state = np.random.binomial(1, p, self.size).astype(int)

    def compute_next_state(self, step=1):

        for i in range(step):

            previous_state = np.copy(self.state)

            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    self.state[i][j] = self.law_book[(
                        previous_state[(i - 1) % self.size[0]][(j - 1) % self.size[1]],
                        previous_state[(i - 1) % self.size[0]][(j - 0) % self.size[1]],
                        previous_state[(i - 1) % self.size[0]][(j + 1) % self.size[1]],
                        previous_state[(i - 0) % self.size[0]][(j - 1) % self.size[1]],
                        previous_state[(i - 0) % self.size[0]][(j - 0) % self.size[1]],
                        previous_state[(i - 0) % self.size[0]][(j + 1) % self.size[1]],
                        previous_state[(i + 1) % self.size[0]][(j - 1) % self.size[1]],
                        previous_state[(i + 1) % self.size[0]][(j + 1) % self.size[1]]
                    )]

    @staticmethod
    def base_convert(number, base, size):

        res = []
        while number > 0:
            res.append(number % base)
            number = int(number / base)

        while len(res) < size:
            res.append(0)

        res.reverse()
        return tuple(res)

        # TODO delete

    @staticmethod
    def save_law_book(law_book, folder_name, file_name):
        with open(folder_name + '/' + file_name + '.pkl', 'wb') as f:
            pickle.dump(law_book, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_law_book(folder_name, file_name):
        with open(folder_name + '/' + file_name + '.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def visualize(society, step_count=1000):

        cmap = matplotlib.colors.ListedColormap(['yellow', 'indigo'])
        bounds = [0, 0.5, 1]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        for i in range(step_count):
            ax.cla()
            ax.imshow(society.state, cmap=cmap, norm=norm)
            society.compute_next_state()
            plt.pause(0.01)


