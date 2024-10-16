import numpy as np
import warnings


class HopfieldNetwork:
    def __init__(self, num_neurons: int):
        self.n = num_neurons
        self.weights = np.zeros((self.n, self.n))

    def train(self, patterns):
        patterns = np.array(patterns)
        assert patterns.ndim >= 2
        assert np.issubdtype(patterns.dtype, np.integer)
        num_patterns = patterns.shape[0]

        for p in patterns:
            p = np.ravel(p)
            assert p.size == self.n
            self.weights += np.outer(p, p)

        np.fill_diagonal(self.weights, 0)
        self.weights /= num_patterns

    def update(self, state):
        new_state = np.sign(np.dot(self.weights, state)).astype(np.int8)
        new_state[new_state == 0] = 1
        return new_state

    def predict(self, state, max_iters=1000):
        state = np.array(state)
        shape = state.shape

        state = state.ravel()
        assert state.size == self.n
        assert np.issubdtype(state.dtype, np.integer)

        converged = False
        iters = 0
        for _ in range(max_iters):
            iters += 1
            new_state = self.update(state)
            if np.array_equal(new_state, state):
                converged = True
                break

            state = new_state

        if not converged:
            warnings.warn('Convergence not achieved.')
        else:
            print(f'Convergence achieved in {iters} iterations.')

        return state.reshape(shape)
