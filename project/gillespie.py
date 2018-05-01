"""Gillespie algorithm for Michaelis-Menten enzyme reaction.

   Copyright (c) 2018, Felix Held
"""
import numpy.random as rnd
import numpy as np

import matplotlib.pyplot as plt

def reactions(mu, states):
    """Executes Michaelis-Menten chemical reactions.

    :Arguments:
        mu : int
            Index of the equation
        states : NumPy array
            Current states of the system (E, S, ES, P)

    :Returns:
        NumPy array : Updated state vector (E, S, ES, P)
    """
    if mu == 1:
        # E + S -> ES
        if states[0] > 0 and states[1] > 0:
            states[0] -= 1
            states[1] -= 1
            states[2] += 1
    elif mu == 2:
        # ES -> E + S
        if states[2] > 0:
            states[0] += 1
            states[1] += 1
            states[2] -= 1
    elif mu == 3:
        # ES -> E + P
        if states[2] > 0:
            states[0] += 1
            states[2] -= 1
            states[3] += 1
    else:
        raise ValueError('Reaction mu = %d does not exist.' % mu)

    return states


if __name__ == '__main__':
    # Rates
    c = np.array([.1, .1, .4])
    # Initial counts
    states = np.array([10000, 1000, 0, 0])
    saved_states = np.zeros((1, 4), dtype='int')
    saved_states[0, :] = states
    # Time
    ts = [0]

    # # Initial state
    # print("Time t = %.2f, States (E = %d, S = %d, ES = %d, P = %d)" %
    #         (t, states[0], states[1], states[2], states[3]))

    # Endtime
    tend = 20

    while ts[-1] < tend and (states[1] > 0 or states[2] > 0):
        # Calculate rate function values
        h = np.array([
            c[0] * states[0] * states[1],
            c[1] * states[2],
            c[2] * states[2]])
        h0 = np.sum(h)

        # Uniform random numbers
        r = rnd.rand(2)
        # Next event time
        tau = np.log(1 / r[0]) / h0
        # Next reaction
        mu = 0
        val = r[1] * h0
        while val > 0.:
            mu += 1
            val -= h[mu - 1]

        # Update time and states
        ts.append(ts[-1] + tau)
        states = reactions(mu, states)
        saved_states = np.concatenate((saved_states, np.array([states])))

        # print("Time t = %.2f, States (E = %d, S = %d, ES = %d, P = %d)" %
        #     (t, states[0], states[1], states[2], states[3]))

    ts = np.array(ts)

    plt.plot(ts, saved_states, 'o', markersize=1)
    plt.legend(['E', 'S', 'ES', 'P'])
    plt.savefig('traj.png')
