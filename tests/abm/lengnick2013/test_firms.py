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

def configure_inventory_level(f, x, level):
    """
    Given desired relationships between inventory and the inventory bounds,
    set appropriate values for demand (d), inventory (i), and inventory
    bounds (i_phi_lower & i_phi_upper).

    Args:
        f (Firms object) - the object holding the firms
        x (int) - the firm to configure
        level (int) - relationship between inventory and the bounds to
           be configured
            -2: i < lb < ub
            -1: i = lb < ub
             0: lb < i < ub
            +1: lb < i = ub
            +2: lb < ub < i
    """

    f.d[x] = 5
    f.i_phi_lower[x] = 0.4 # 5 * 0.2 = 2
    f.i_phi_upper[x] = 1.2 # 5 * 1.2 = 6

    f.i[x] = (level < 0)  * f.d[x] * f.i_phi_lower[x] + \
             (level >= 0) * f.d[x] * f.i_phi_upper[x] + \
             (level in [-2, 0]) * (-1) + \
             (level == 2) * (+1)

def configure_vacancy_free_level(f, x, level):
    """
    Given desired relationships between vacancy-free months and the
    vacancy-free threshhold, set appropriate values for vacancy-free (nv),
    vacancy-free threshhold (gamma).

    Args:
        f (Firms object) - the object holding the firms
        x (int) - the firm to configure
        level (int) - relationship between vacancy-free and the threshhold to
           be configured
             0: vacancy-free months equal to zero
            -1: vacancy-free months greater than zero but less than threshhold
            +1: vacancy-free months equal to threshhold
            +2: vacancy-free months greather than threshhold
    """

    f.gamma[x] = 5

    f.nv[x] = (level == -1) * (f.gamma[x] - 1) + \
              (level == +1) * f.gamma[x] + \
              (level == +2) * (f.gamma[x] + 1)

def test_adjust_wages():

    # Test Cases:
    #
    # Notes:
    # - vacancy-free count is for last month (doesn't include "next month")
    # - vacancy-free can never be (>= thresh) AND (NOT > 0) therefore these
    #   cases are not included in the set of test cases
    #
    # vacancy-free last month:
    # - 0: zero vacancy free months
    # - <: vacancy-free months > 0 AND < threshhold (gamma)
    # - =: vacancy-free months = threshold (gamma)
    # - >: vacancy-free months > threshold (gamma)
    #
    # Case #                              00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    # INPUT STATE:
    # inventory (<l/=l/>l)                <l <l <l <l <l <l <l <l =l =l =l =l =l =l =l =l >l >l >l >l >l >l >l >l
    # vacancy-free - last month (0/</=/>)  0  0  <  <  =  =  >  >  0  0  <  <  =  =  >  >  0  0  <  <  =  =  >  >
    # vacancy open - last month (T/F)      F  T  F  T  F  T  F  T  F  T  F  T  F  T  F  T  F  T  F  T  F  T  F  T
    # OUTPUT/ACTIONS:                               *     *     *           *     *     *           *     *     *
    # raise exception (T/F)                F  F  F  T  F  T  F  T  F  F  F  T  F  T  F  T  F  F  F  T  F  T  F  T
    # vacancy open - next month (T/F)      T  T  T  T  T  T  T  T  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F
    # vacancy-free - next month (0/+)      0  0  0  0  0  0  0  0  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +  +
    # wage change (+/0/-)                  0  +  0  E  0  E  0  E  0  0  0  E  -  E  -  E  0  0  0  0  -  E  -  E

    N = 24

    # configure parameters for the test firms
    inv_levels                = np.array([-2,-2,-2,-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0,+1,+1,+2,+2])
    vacancy_free_level        = np.array([ 0, 0,-1,-1,+1,+1,+2,+2, 0, 0,-1,-1,+1,+1,+2,+2, 0, 0,-1,-1,+1,+1,+2,+2])
    vacancy_open_last_month   = np.array([ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    # delta: wage % change upper bound
    wage_delta                = np.full(N, 0.1)
    # wages "last month"
    wage = old_wage           = np.full(N, 5)

    # define results
    # vacancy-open next month: not currently tested (because adjust_wages doesn't modify vacancies)
    vacancy_open_next_month   = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # vacancy-free change: not currently tested (because adjust_wages doesn't modify vacancy-free months)
    #    0: set to zero
    #   +1: add one
    vacancy_free_change       = np.array([ 0, 0, 0, 0, 0, 0, 0, 0,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1])
    # expected wage change
    #  +1: wage increases
    #   0: wage remains unchanged
    #  -1: wage decreases
    #  -9: exception
    wage_change               = np.array([ 0,+1, 0,-9, 0,-9, 0,-9, 0, 0, 0,-9,-1,-9,-1,-9, 0, 0, 0,-9,-1,-9,-1,-9])
    # error cases
    exception_expected = (wage_change == -9)

    for x in range(N):
        print(x)
        f = firms.Firms(1)
        # configure inventory level (inventory, demand, & bounds parameters)
        configure_inventory_level(f, 0, inv_levels[x])
        # vacancy last month
        f.v[0] = vacancy_open_last_month[x]
        # configure vacancy-free level (vacancy-free months & threshhold)
        configure_vacancy_free_level(f, 0, vacancy_free_level[x])
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

    # Case #                          00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19
    # INPUT STATE:
    # inventory (<l/=l/0/=u/>u)       <l <l <l <l =l =l =l =l  0  0  0  0 =u =u =u =u >u >u >u >u
    # vacancy open - last month (T/F)  F  F  T  T  F  F  T  T  F  F  T  T  F  F  T  T  F  F  T  T
    # zero workforce (T/F)             F  T  F  T  F  T  F  T  F  T  F  T  F  T  F  T  F  T  F  T
    # OUTPUT/ACTIONS:
    # vacancy open - next month (T/F)  T  T  T  T  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F
    # workforce decrease (T/F)         F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  T  F  T  F

    N = 20

    # configure parameters for the test firms
    inv_levels            = np.array([-2,-2,-2,-2,-1,-1,-1,-1, 0, 0, 0, 0,+1,+1,+1,+1,+2,+2,+2,+2])
    vacancy               = np.array([ 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    workforce_size        = np.array([ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    # define results
    new_vacancy           = np.array([ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    workforce_change      = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,-1, 0])

    for x in range(N):
        f = firms.Firms(1)
        # configure inventory level (inventory, demand, & bounds parameters)
        configure_inventory_level(f, 0, inv_levels[x])
        # configure vacancy
        f.v[0] = vacancy[x]
        # configure workforce size
        f.l[0] = workforce_size[x]
        # remember workforce size

        # adjust workforce
        f.adjust_workforce()

        # check result
        assert f.v[0] == new_vacancy[x]
        assert f.l[0] == (workforce_size[x] + workforce_change[x])

def test_price_change_type():
    '''
    Test that, given a particular inventory level, the correct type of
    price change occurs.
    '''
    # Test Cases:
    # Case #                      00   01   02   03   04
    # INPUT STATE:
    # inventory (<l/=l/0/=u/>u)  <lb  =lb  000  =ub  >ub
    # OUTPUT/ACTIONS:
    # price change (-/0/+)        -    0    0    0    +

    N = 5

    # configure parameters for the test firms
    inv_levels       = np.array([ -2,  -1,   0,  +1,  +2])

    # define results
    price_change     =np.array([  +1,   0,   0,   0,  -1])

    for x in range(N):
        f = firms.Firms(1)
        # configure inventory level (inventory, demand, & bounds parameters)
        configure_inventory_level(f, 0, inv_levels[x])
        # configure price max change %
        f.nu[0] = 0.05
        # configure probability of accepting price change
        # - set to 100% to ensure that price change occurs
        f.theta[0] = 1
        # configure current price and remember for results checking
        f.p[0] = old_p = 1

        # configure marginal cost to be equal to current price
        # -- marginal cost: f.w / f.t_lambda
        f.w[0] = 1
        f.t_lambda[0] = 1

        # configure wide price bounds so they don't play a role in
        # deteremining the price change
        f.p_phi_upper[0] = 100 # new price can be 100 times as great as current price
        f.p_phi_lower[0] = 0   # new price can be zero

        # adjust workforce
        f.adjust_prices()

        # check result
        if (price_change[x] < 0):
            assert f.p[0] < old_p
        elif (price_change[x] == 0):
            assert f.p[0] == old_p
        else: # (price_change > 0)
            assert f.p[0] > old_p

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