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
    # - wage was decreased
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

def test_adjust_workforce():
    f = firms.Firms(7)

    # i_phi_upper: max % of previous month demand
    f.i_phi_upper = np.full(f.F, 1.0)

    # i_phi_lower: min % of previous month demand
    f.i_phi_lower = np.full(f.F, 0.25)

    # configure previous month demand
    f.d = np.array([1, 2, 5, 6, 20, 25, 30])

    # configure current inventory
    f.i = np.full(f.F, 5)

    # configure vacancies
    f.v = np.array([0, 1, 0, 2, 0, 0, 1])

    # configure workforce sizes
    f.l = np.array([0, 1, 2, 3, 4, 5, 6])

    # save current workforce sizes for comparison
    l_old = f.l

    # save current vacancies for comparison
    v_old = f.v

    # adjust workforce
    f.adjust_workforce()

    # test firm 1: inventory above upper limit but workforce min size
    # - workforce remained unchanged at zero
    assert f.l[0] == l_old[0]
    assert f.l[0] == 0
    # - no vacancies
    assert f.v[0] == 0

    # test firm 2: inventory above upper limit
    # - workforce size is one less
    assert f.l[1] == (l_old[1] - 1)
    # - workforce size is >= workforce min size
    assert f.l[1] >= 0
    # - no vacancies exist
    assert f.v[1] == 0

    # test firm 3: inventory equal to upper limit
    # - workforce size is unchanged
    assert f.l[2] == l_old[2]
    # - no vacancies exist
    assert f.v[2] == 0

    # test firm 4: inventory between limits
    # - workforce size is unchanged
    assert f.l[3] == l_old[3]
    # - no vacancies exist
    assert f.v[3] == 0

    # test firm 5: inventory equal to lower limit
    # - workforce size is unchanged
    assert f.l[4] == l_old[4]
    # - no vacancies exist
    assert f.v[4] == 0

    # test firm 6: inventory below lower limit w/no pre-existing vacancy
    # - workforce size is unchanged
    assert f.l[5] == l_old[5]
    # - one new vacancy exists
    assert v_old[5] == 0 # validate test data
    assert f.v[5] == 1

    # test firm 7: inventory below lower limit w/pre-existing vacancy
    # - workforce size is unchanged
    assert f.l[6] == l_old[6]
    # - previously existing vacancy still exists
    assert v_old[6] == 1 # validate test data
    assert f.v[6] == 1