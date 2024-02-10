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

def test_adjust_prices():

    # Test Cases:
    #
    # Unless noted otherwise, for the following test cases,
    # theta is set to 1.0 to ensure that price change occurs
    #
    # Case    Inventory     Calculated Price   New Price
    # ---- ---------------  ----------------  -----------
    #  1a    i < lb < ub      p < lb < ub         lb
    #  1b    i < lb < ub      lb = p < ub         lb
    #  1c    i < lb < ub      lb < p < ub         p
    #  1d    i < lb < ub      lb < p = ub         ub
    #  1e    i < lb < ub      lb < ub < p         ub
    #
    #  2a    lb = i < ub      p < lb < ub      no change
    #  2b    lb = i < ub      lb = p < ub      no change
    #  2c    lb = i < ub      lb < p < ub      no change
    #  2d    lb = i < ub      lb < p = ub      no change
    #  2e    lb = i < ub      lb < ub < p      no change
    #
    #  3a    lb < i < ub      p < lb < ub      no change
    #  3b    lb < i < ub      lb = p < ub      no change
    #  3c    lb < i < ub      lb < p < ub      no change
    #  3d    lb < i < ub      lb < p = ub      no change
    #  3e    lb < i < ub      lb < ub < p      no change
    #
    #  4a    lb < i = ub      p < lb < ub      no change
    #  4b    lb < i = ub      lb = p < ub      no change
    #  4c    lb < i = ub      lb < p < ub      no change
    #  4d    lb < i = ub      lb < p = ub      no change
    #  4e    lb < i = ub      lb < ub < p      no change
    #
    #  5a    lb < ub < i      p < lb < ub         lb
    #  5b    lb < ub < i      lb = p < ub         lb
    #  5c    lb < ub < i      lb < p < ub         p
    #  5d    lb < ub < i      lb < p = ub         ub
    #  5e    lb < ub < i      lb < ub < p         ub

    # initialize firms
    f = firms.Firms(25)

    # configure constant parameters for the test firms
    # - configure current inventory
    f.i = np.full(f.F, 5)
    # - configure previous month demand
    f.d = np.full(f.F, 5)
    # - configure current wages
    f.w = np.full(f.F, 5)
    # - configure worker productivity (t_lambda)
    f.t_lambda = np.full(f.F, 1)
    # - configure current price (equal to marginal cost)
    mc = f.w * f.t_lambda
    f.p = np.full(f.F, mc)
    # set price change probability to one
    # - so that price changes are always accepted
    f.theta = np.full(f.F, 1.0)

    # initialize inventory bounds factors
    i_lower = [1.1, 1.0, 0.9, 0.8, 0.7]
    i_upper = [1.3, 1.2, 1.1, 1.0, 0.9]

    # initialize price bounds factors
    p_lower = [1.1, 1.0, 0.9, 0.8, 0.7]
    p_upper = [1.3, 1.2, 1.1, 1.0, 0.9]

    # initialize price change factors(nu)
    p_nu = [0.1, 0.0, 0.1, 0.0, 0.1]

    # configure variable parameters for the test firms
    for x in range(25):
        # configure inventory bounds factors:
        f.i_phi_lower[x] = i_lower[x//5]
        f.i_phi_upper[x] = i_upper[x//5]

        # test inventory bounds
        if (x//5 == 0):
            assert (f.i[x] < f.i_phi_lower[x]*f.d[x])  and (f.i_phi_lower[x]*f.d[x] < f.i_phi_upper[x]*f.d[x])
        elif (x//5 == 1):
            assert (f.i_phi_lower[x]*f.d[x] == f.i[x]) and (f.i[x] < f.i_phi_upper[x]*f.d[x])
        elif (x//5 == 2):
            assert (f.i_phi_lower[x]*f.d[x] < f.i[x])  and (f.i[x] < f.i_phi_upper[x]*f.d[x])
        elif (x//5 == 3):
            assert (f.i_phi_lower[x]*f.d[x] < f.i[x])  and (f.i[x] == f.i_phi_upper[x]*f.d[x])
        else: # (x//5 == 4)
            assert (f.i_phi_lower[x]*f.d[x] < f.i[x])  and (f.i_phi_upper[x]*f.d[x] < f.i[x])

        # configure price bounds factors
        f.p_phi_lower[x] = p_lower[x%5]
        f.p_phi_upper[x] = p_upper[x%5]

        # test price bounds
        if (x%5 == 0):
            assert (f.p[x] < f.p_phi_lower[x]*mc[x])  and (f.p_phi_lower[x]*mc[x] < f.p_phi_upper[x]*mc[x])
        elif (x%5 == 1):
            assert (f.p_phi_lower[x]*mc[x] == f.p[x]) and (f.p[x] < f.p_phi_upper[x]*mc[x])
        elif (x%5 == 2):
            assert (f.p_phi_lower[x]*mc[x] < f.p[x])  and (f.p[x] < f.p_phi_upper[x]*mc[x])
        elif (x%5 == 3):
            assert (f.p_phi_lower[x]*mc[x] < f.p[x])  and (f.p[x] == f.p_phi_upper[x]*mc[x])
        else: # (x%5 == 4):
            assert (f.p_phi_lower[x]*mc[x] < f.p[x])  and (f.p_phi_upper[x]*mc[x] < f.p[x])

        # configure price change factor
        f.nu[x] = p_nu[x%5]

        # test price change factor
        if (x%5 == 0) or (x%5 == 2) or (x%5 == 4):
            assert f.p[x] < (f.p[x] + np.random.uniform(0, f.nu[x]))
        else:
            assert f.p[x] == (f.p[x] + np.random.uniform(0, f.nu[x]))

    # save old prices
    old_p = f.p

    # test adjust_prices()
    f.adjust_prices()

    # evaluate result\
    p_lower_bound = f.p_phi_lower*mc
    p_upper_bound = f.p_phi_upper*mc

    assert f.p[0] == p_lower_bound[0]
    assert f.p[1] == p_lower_bound[1]
    assert (p_lower_bound[2] < f.p[2]) and (f.p[2] > old_p[2]) and (f.p[2] < p_upper_bound[2])
    assert f.p[3] == p_upper_bound[3]
    assert f.p[4] == p_upper_bound[4]

    # TODO: complete test case
