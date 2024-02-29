import pytest
import traceback

import numpy as np
import abm.lengnick2013.firms as firms

def test_set_prop():

    # Test Cases:
    #
    #    case  propName    id           value             result
    #    ----  --------   -----   ------------------  ---------------
    #     1       -         -             -              TypeError
    #     2       1         -             1              TypeError
    #     3      foo        -             -              TypeError
    #     4     gamma       -           "foo"            TypeError
    #     5      foo        -             1            AttributeError
    #     6     gamma       -         2D array           TypeError
    #     7     gamma       -     array wrong length     TypeError
    #     8     gamma     "foo"           1              TypeError
    #     9     gamma       1           array            TypeError
    #    10     gamma      -1             1              IndexError
    #    11     gamma       5             1              IndexError
    #    12     gamma       -             1            gamma all 1's
    #    13     gamma       -        [1,2,3,4,5]       gamma = array
    #    14     gamma       2             3             f.gamma[2]=3

    # initialize test firms
    f = firms.Firms(5)

    # cases 1, 2, 3, & 4
    with pytest.raises(TypeError):
        f.set_prop()
        f.set_prop(1)
        f.set_prop("foo")

    # case 5
    with pytest.raises(AttributeError):
        f.set_prop("foo", 1)

    # cases 6, 7, 8, & 9
    with pytest.raises(TypeError):
        f.set_prop("gamma", np.array([1,2,3], [1,2,3]))
        f.set_prop("gamma", np.array([1,2,3]))
        f.set_prop("gamma", "foo", id=1)
        f.set_prop("gamma", np.full(f.F, 1.1), id=1)

    # cases 10 & 11
    with pytest.raises(IndexError):
        f.set_prop("gamma", 1, id=-1)
        f.set_prop("gamma", 1, id=f.F)

    # case 12
    f.set_prop("gamma", 1)
    assert np.array_equal(f.gamma, np.full(f.F, 1))

    # case 13
    new_gamma = np.array([1,2,3,4,5])
    f.set_prop("gamma", new_gamma)
    assert np.array_equal(f.gamma, new_gamma)

    # case 14
    old_gamma = f.gamma
    f.set_prop("gamma", 3, id=2)
    assert f.gamma[2] == 3
    assert np.array_equal(f.gamma[:2], old_gamma[:2])
    assert np.array_equal(f.gamma[3:], old_gamma[3:])

def test_adjust_wages():

    # Test Cases:
    #
    # Notes:
    # - vacancy-free count is for last month (doesn't include "next month")
    # - Vacancy-free can never be (>= thresh) AND (NOT > 0)
    #
    # Case #                          00 01 02 03 04 05 06 07 08 09 10 11
    # INPUT STATE:
    # vacancy open - last month (T/F)  F  F  F  F  F  F  T  T  T  T  T  T
    # vacancy open - next month (T/F)  F  F  F  T  T  T  F  F  F  T  T  T
    # vacancy-free > 0 (T/F)           F  T  T  F  T  T  F  T  T  F  T  T
    # vacancy-free >= thresh (T/F)     F  F  T  F  F  T  F  F  T  F  F  T
    # OUTPUT/ACTIONS:                                       *  *     *  *
    # raise exception (T/F)            F  F  F  F  F  F  F  T  T  F  T  T
    # wage change (+/0/-)              0  0  -  0  0  0  0  E  E  +  E  E

    N = 12

    # vacancy opened "last month"
    vacancy_last_month = np.array([ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 ])

    # vacancy opened "next month"
    # - vacancy is opned next month if inventory is below the
    #   lower bound defined by the following:
    #   f.i_phi_lower * f.d = 0.2 * 5 = 1
    demand = np.full(N, 5)
    i_phi_lower = np.full(N, 0.2)
    # some cases set equal to threshhold (1) others above threshhold (2, 3)
    inventory  = np.array([ 1, 2, 3, 0, 0, 0, 1, 2, 3, 0, 0, 0 ])

    # vacancy-free months threshold
    vacancy_free_thresh = np.full(N, 5)
    vacancy_free_months = np.array([0, 1, 5, 0, 2, 6, 0, 3, 7, 0, 4, 8])

    # results: expected wage changes
    #  +1: wage increases
    #   0: wage remains unchanged
    #  -1: wage decreases
    #  -9: exception
    wage_change = np.array([ 0, 0,-1, 0, 0, 0, 0,-9,-9,+1,-9,-9 ])

    # results: error cases
    exception_expected = (wage_change == -9)

    # delta: wage % change upper bound
    wage_delta = np.full(N, 0.1)

    # wages "last month"
    wage = old_wage = np.full(N, 5)

    # iterate through test cases
    for x in range(N):
        f = firms.Firms(1)
        # vacancy last month
        f.v[0] = vacancy_last_month[x]
        # vacancy next month
        f.d[0] = demand[x]
        f.i_phi_lower[0] = i_phi_lower[x]
        f.i[0] = inventory[x]
        # vacancy-free months
        f.gamma[0] = vacancy_free_thresh[x]
        f.nv[0] = vacancy_free_months[x]
        # wage & delta
        f.delta[0] = wage_delta[x]
        f.w[0] = wage[x]

        try:
            f.adjust_wages()
        except Exception as e:
            assert exception_expected[x]
        else:
            if (wage_change[x] > 0):
                assert old_wage[x] < f.w[0]
            elif (wage_change[x] < 0):
                assert old_wage[x] > f.w[0]
            else: # (wage_change[x] == 0)
                assert old_wage[x] == f.w[0]

def test_adjust_workforce():

    # Test Cases:
    #
    # Case    Inventory      Vacancy    Workforce   New Vacancy  New Workforce
    # ---- ---------------  ---------  -----------  -----------  -------------
    #  1a    i < lb < ub        0           1            1         unchanged
    #  1b    i < lb < ub        1           1        unchanged     unchanged
    #  2a    lb = i < ub        0           1        unchanged     unchanged
    #  2b    lb = i < ub        1           1            0         unchanged
    #  3a    lb < i < ub        0           1        unchanged     unchanged
    #  3b    lb < i < ub        1           1            0         unchanged
    #  4a    lb < i = ub        0           1        unchanged     unchanged
    #  4b    lb < i = ub        1           1            0         unchanged
    #  5a    lb < ub < i        0           1        unchanged         0
    #  5b    lb < ub < i        1           1            0             0
    #  5c    lb < ub < i        0           0        unchnaged     unchanged
    #  5d    lb < ub < i        1           0            0         unchanged

    N = 12
    f = firms.Firms(N)

    # configure cparameters for the test firms
    # - configure current inventory
    f.i = np.full(N, 5)
    # - configure previous month demand
    f.d = np.full(N, 5)
    # configure vacancies
    f.v = np.array([x%2 for x in range(N)])
    # configure workforce sizes
    f.l = np.array([1*(x<10) for x in range(N)])

    # initialize inventory bounds factors
    f.i_phi_lower = np.array([1.1, 1.1, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7])
    f.i_phi_upper = np.array([1.3, 1.3, 1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9])

    # new vacancy - result array
    new_v = f.v
    new_v[0] = 1
    new_v[3] = new_v[5] = new_v[7] = new_v[9] = new_v[11] = 0
    # new workforce - result array
    new_l = f.l
    new_l[8] = new_l[9] = 0

    # save current vacancies for comparison
    v_old = f.v

    # save current workforce for comparison
    l_old = f.l

    # adjust workforce
    f.adjust_workforce()

    # verify result
    for x in range(12):
        assert f.l[x] == new_l[x]
        assert f.v[x] == new_v[x]

def test_adjust_prices():

    # Test Cases:
    #
    # Note: For cases 1 - 5, theta is set to 1.0 to ensure
    # that price change is always accepted.
    #
    # Note: The price change factor (nu) is set such that
    # if the current price is outside the price bounds the 
    # new price will still be outside the bounds.  In these
    # cases, the new price will be set to the appropriate
    # price bound (see cases 1a & 5e).
    #
    # Note: If a price increase is required, and the current
    # price is above the price upper bound, the price is NOT
    # decreased to bring it within the bounds (see case 1e).
    # Similarly, if a price decrease is required, and the
    # price is below the price lower bound, the price is NOT
    # increased to bring it within the bounds (see case 5a).
    #
    # Case    Inventory     Calculated Price   New Price
    # ---- ---------------  ----------------  -----------
    #  1a    i < lb < ub      p < lb < ub         lb          (***)
    #  1b    i < lb < ub      lb = p < ub         lb
    #  1c    i < lb < ub      lb < p < ub      old_p < p
    #  1d    i < lb < ub      lb < p = ub         ub
    #  1e    i < lb < ub      lb < ub < p      p == old_p     (***)
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
    #  5a    lb < ub < i      p < lb < ub      p == old_p    (***)
    #  5b    lb < ub < i      lb = p < ub         lb
    #  5c    lb < ub < i      lb < p < ub      p < old_p
    #  5d    lb < ub < i      lb < p = ub         ub
    #  5e    lb < ub < i      lb < ub < p         ub         (***)
    #
    # Note: For case 6, theta is set to zero so that the
    # price change is not accepted.
    #
    #  6a    i < lb < ub      lb < p < ub      old_p == p
    #  6b    lb < ub < i      lb < p < ub      old_p == p

    # initialize firms
    N = 27
    f = firms.Firms(N)

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

    # initialize inventory bounds factors
    i_lower = [1.1, 1.0, 0.9, 0.8, 0.7]
    i_upper = [1.3, 1.2, 1.1, 1.0, 0.9]

    # initialize price bounds factors
    p_lower = [1.1, 1.0, 0.9, 0.8, 0.7]
    p_upper = [1.3, 1.2, 1.1, 1.0, 0.9]

    # initialize price change factors (nu)
    p_nu = [0.1, 0.0, 0.1, 0.0, 0.1]

    # configure variable parameters for the test firms
    for x in range(N):
        # configure inventory bounds factors:
        if (x//5 < 5):
            f.i_phi_lower[x] = i_lower[x//5]
            f.i_phi_upper[x] = i_upper[x//5]
        else: # (x//5 == 5) - cases 6a & 6b (firms 25 & 26)
            f.i_phi_lower[x] = i_lower[(x%5) * 4] # (25%5 == 0), (26%5 == 1)
            f.i_phi_upper[x] = i_upper[(x%5) * 4]

        # test inventory bounds
        if (x//5 == 0):
            assert (f.i[x] < f.i_phi_lower[x]*f.d[x])  and (f.i_phi_lower[x]*f.d[x] < f.i_phi_upper[x]*f.d[x])
        elif (x//5 == 1):
            assert (f.i_phi_lower[x]*f.d[x] == f.i[x]) and (f.i[x] < f.i_phi_upper[x]*f.d[x])
        elif (x//5 == 2):
            assert (f.i_phi_lower[x]*f.d[x] < f.i[x])  and (f.i[x] < f.i_phi_upper[x]*f.d[x])
        elif (x//5 == 3):
            assert (f.i_phi_lower[x]*f.d[x] < f.i[x])  and (f.i[x] == f.i_phi_upper[x]*f.d[x])
        elif (x//5 == 4):
            assert (f.i_phi_lower[x]*f.d[x] < f.i[x])  and (f.i_phi_upper[x]*f.d[x] < f.i[x])
        else: # (x//5 == 5) - cases 6a & 6b (firms 25 & 26)
            if (x == f.F-2): # firm 25
                assert (f.i[x] < f.i_phi_lower[x]*f.d[x])  and (f.i_phi_lower[x]*f.d[x] < f.i_phi_upper[x]*f.d[x])
            else: # firm 26
                assert (f.i_phi_lower[x]*f.d[x] < f.i[x])  and (f.i_phi_upper[x]*f.d[x] < f.i[x])

        # configure price bounds factors
        if (x//5 < 5):
            f.p_phi_lower[x] = p_lower[x%5]
            f.p_phi_upper[x] = p_upper[x%5]
        else: # (x//5 == 5) - cases 6a & 6b (firms 25 & 26)
            f.p_phi_lower[x] = p_lower[2]
            f.p_phi_upper[x] = p_upper[2]

        # test price bounds
        if (x%5 == 0) and (x//5 < 5): # don't include cases 6a & 6b
            assert (f.p[x] < f.p_phi_lower[x]*mc[x])  and (f.p_phi_lower[x]*mc[x] < f.p_phi_upper[x]*mc[x])
        elif (x%5 == 1) and (x//5 < 5): # don't include cases 6a & 6b
            assert (f.p_phi_lower[x]*mc[x] == f.p[x]) and (f.p[x] < f.p_phi_upper[x]*mc[x])
        elif (x%5 == 2) or (x//5 == 5):  # include cases 6a & 6b
            assert (f.p_phi_lower[x]*mc[x] < f.p[x])  and (f.p[x] < f.p_phi_upper[x]*mc[x])
        elif (x%5 == 3):
            assert (f.p_phi_lower[x]*mc[x] < f.p[x])  and (f.p[x] == f.p_phi_upper[x]*mc[x])
        else: # (x%5 == 4)
            assert (f.p_phi_lower[x]*mc[x] < f.p[x])  and (f.p_phi_upper[x]*mc[x] < f.p[x])

        # configure price change factor
        if (x//5 < 5):
            f.nu[x] = p_nu[x%5]
        else:  # (x//5 == 5) - cases 6a & 6b (firms 25 & 26)
            f.nu[x] = p_nu[2]

        # test price change factor
        if (x%5 in (0, 2, 4)) or (x//5 == 5): # include cases 6a & 6b
            assert f.p[x] < (f.p[x] + np.random.uniform(0, f.nu[x]))
        else: # (x%5 in (1, 3)):
            assert f.p[x] == (f.p[x] + np.random.uniform(0, f.nu[x]))

        # configure price change probability
        if (x//5 < 5):
            # - set to 1.0 so price always changes
            f.theta[x] = 1.0
        else: # (x//5 == 5) - cases 6a & 6b (firms 25 & 26)
            # - set to 0.0 so price never changes
            f.theta[x] = 0.0

    # save old prices
    old_p = f.p

    # test adjust_prices()
    f.adjust_prices()

    # calculate bounds - for use in result evaluation
    p_lower_bound = f.p_phi_lower*mc
    p_upper_bound = f.p_phi_upper*mc

    # evaluate result
    for x in range(N):
        if (x//5 in (1, 2, 3, 5)): # price doesn't change
            assert (f.p[x] == old_p[x])
        elif (x//5 == 0):
            if (x%5 in (0, 1)):
                assert (f.p[x] == p_lower_bound[x])
            elif (x%5 == 2):
                assert (p_lower_bound[x] < f.p[x]) and (f.p[x] < p_upper_bound[x])
                assert (old_p[x] < f.p[x])
            elif (x%5 == 3):
                assert (f.p[3] == p_upper_bound[3])
            elif (x%5 == 4):
                assert (f.p[x] == old_p[x]) # will fail
            else:
                assert False # unexpected case
        elif (x//5 == 4):
            if (x%5 == 0):
                assert (f.p[x] == old_p[x]) # will fail
            elif (x%5 == 1):
                assert (f.p[x] == p_lower_bound[x])
            elif (x%5 == 2):
                assert (p_lower_bound[x] < f.p[x]) and (f.p[x] < p_upper_bound[x])
                assert (f.p[x] < old_p[x])
            elif (x%5 in (3, 4)):
                assert (f.p[3] == p_upper_bound[3])
            else:
                assert False # unexpected case
        else:
            assert False # unexpected case