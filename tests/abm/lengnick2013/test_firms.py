import numpy as np
import abm.lengnick2013.firms as firms

def test_adjust_wages():
    f = firms.Firms(5)

    # delta: wage % change upper bound
    f.delta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # gamma: no. of vacancy-free months before reducing wages
    f.gamma = np.array([23, 24, 24, 24, 25])

    # configure vacancies
    f.v = np.array([0, 1, 0, 2, 0])

    # configure months w/o vacancy
    f.nv = np.array([24, 0, 24, 0, 24])
 
    # configure initial wages
    f.w = np.ones(f.F)

    # save current wages for comparison
    w_old = f.w

    # adjust wages
    f.adjust_wages()

    # test firm 1: vacancy-free for more than gamma months
    # - wage was decreases
    assert f.w[0] < w_old[0]
    # - wage decrease was less than delta %
    assert f.w[0] >= (w_old[0] * (1-f.delta[0]))

    # test firm 2: has 1 vacancy remaining
    # - wage was increased
    assert f.w[1] > w_old[1]
    # - wage increase was less than delta %
    assert f.w[1] <= (w_old[1] * (1+f.delta[1]))

    # test firm 3: vacancy-free for exactly gamma months
    # - wage was unchanged
    assert f.w[2] == w_old[2]

    # test firm 4: has 2 vacancies remaining
    # - wage was increased
    assert f.w[3] > w_old[3]
    # - wage increase was less than delta %
    assert f.w[3] <= (w_old[3] * (1+f.delta[3]))

    # test firm 5: vacancy-free for less than gamma months
    # - wage was unchanged
    assert f.w[4] == w_old[4]
