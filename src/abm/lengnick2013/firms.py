import numpy as np
import numbers

class Firms:
    # initialize firms
    def __init__(self, F):

        # initialize number of firms
        self.F = F
        
        # initialize model parameters
        # - See [Lengnick 2013] Table 1:
        # firm number of vacancy-free months before reducing wage rate (gamma_f)
        self.gamma = np.full(self.F, 24) # [Lengnick 2013] sets this to 24
        # firm wage max % change (delta_f)
        self.delta = np.full(self.F,0.019) # [Lengnick 2013] sets this to 0.019
        # TODO: store inventory bounds as a 2-column matrix to allow both
        #       bounds to be calculated using matrix multiplication
        # firm inventory upper bound - percentage of previous demand
        self.i_phi_upper = np.full(self.F, 1.0)  # [Lengnick 2013] sets this to 1
        # firm inventory lower bound - percentage of previous demand
        self.i_phi_lower = np.full(self.F, 0.25)  # [Lengnick 2013] sets this to 0.25
        # firm price max % change (nu_f)
        self.nu = np.full(self.F, 0.02)  # [Lengnick 2013] sets this to 0.02
        # TODO: store price bounds as a 2-column matrix to allow both
        #       bounds to be calculated using matrix multiplication
        # firm price upper bound - percentage of marginal costs
        self.p_phi_upper = np.full(self.F, 1.15)  # [Lengnick 2013] sets this to 1.15
        # firm price lower bound - percentage of marginal costs
        self.p_phi_lower = np.full(self.F, 1.025)  # [Lengnick 2013] sets this to 1.025
        # firm probability of accepting price change (theta_f)
        self.theta = np.full(self.F, 0.75) # [Lengnick 2013] sets this to 0.75
        # firm technology level - units produced per employee
        self.t_lambda = np.full(self.F, 3) # [Lengnick 2013] sets this to 3

        # initial conditions (TBD)
        # firm liquidity (m_f) - current "bank account" balance
        self.m = np.zeros(self.F) # bank balance is zero at start (?)
        # firm inventory (i_f) - current inventory levels
        self.i = np.full(self.F, 5) # inventory set to 5 at start (?)
        # firm previous demand (d_f) - the demand for the previous month
        self.d = np.full(self.F, 5) # set demand equal to inventory at start (?)
        # firm wage (w_f) - current wage paid to employees
        self.w = np.ones(self.F) # wage set to 1 at start (?)
        # firm price (p_f) - current price
        self.p = np.ones(self.F) # price set to 1 at start (?)
        # firm labor (l_f) - number of households currently employed by each firm
        self.l = np.ones(self.F, dtype=int) # employees set to 1 at start (?)
        # firm vacancies - current open positions
        self.v = np.zeros(self.F, dtype=int) # every firm has zero vacancy at start
        # firm number of months without vacancy
        self.nv = np.zeros(self.F, dtype=int) # no months w/o vacancy at start

    def set_prop(self, prop_name, value, id=None):
        '''
        Set the value of the firm property either for a single firm or
        for all firms.

        Args:
            prop_name (string):The name of the property to update
            id (int, optional):
                 The `id` of the firm whose property should be updated. If
                 a value is specified for `id` then `array` must be equal
                 to `None`.
            value (numbers.Number of numpy.ndarray):
                 The value to be set for the specified property.  If a value
                 for `id` is provided then `value` must be a `numbers.Number`.
                 If `id` is equal to `None` then the specified `value` s used
                 to update the values of all the firms.  In this case, if a
                 single `numbes.Number` value is provided, the property for
                 all firms is set to the specified `value`.  If a `numpy.Array`
                 is provided then the number of elements in the array must equal
                 the number of firms and the array is used to set property
                 valuse for all the firms.
        '''
        # verify 'prop_name' is valid
        if (not isinstance(prop_name, str)):
            raise TypeError(f"'prop_name' must be a string, not '{type(prop_name).__name__}'")
        if (not hasattr(self, prop_name)):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{prop_name}'")

        # verify that 'value' is valid
        # TODO: allow the array argument to be "array-like" instead of a numpy array
        if not (isinstance(value, numbers.Number) or isinstance(value, np.ndarray)):
            raise TypeError(f"Argument 'value' must be a single numeric value or a 1-dimensional array of length {self.F}.")

        # attempt to set the value of the property
        if (id is None):
            if isinstance(value, numbers.Number):
                setattr(self, prop_name, np.full(self.F, value))
            elif isinstance(value, np.ndarray):
                if (value.ndim == 1 and len(value) == self.F):
                    setattr(self, prop_name, np.full(self.F, value))
                else:
                    raise TypeError(f"Argument 'value' array must be 1-dimensional with length {self.F}.")
            else:
                raise RuntimeError(f"Expected 'value' to be a single numeric value or an array")
        else: # (id is not None)
            # verify that id is an integer
            if (not isinstance(id, int)):
                raise TypeError(f"'id' must be an int, not '{type(id).__name__}'")
            # verify that id is in bounds
            if (id < 0 or id >= self.F):
                raise IndexError(f"'id' {id} is out of bounds: 0 <= 'id' < {self.F}")
            if isinstance(value, numbers.Number):
                arr = getattr(self, prop_name)
                arr[id] = value
            else:
                raise TypeError(f"Argument 'value' must be a single numeric value")

    def adjust_wages(self):
        '''
        See section 2.2 of [Lengnick 2012] for details:

        - Increase wage if vacancy was not filled during previous month
          AND vacancy will remain open during the next month
          condition: (v > 0)
        - Decrease wage if no vacancies for last gamma months
          AND no vacancy will be opened next month
          condition: (nv > g)
        - Randomly select wage change with upper limit equal to delta
          np.random.uniform(0, delta, F)
        - Deteremine direction of wage change (increase or decrease):
          conditions:
          - (v > 0) and (new_v > 0)        -> increase wage (+1)
          - (nv >= gamma) and (new_v == 0) -> decrease wage (-1)
          - otherwise                      -> wage unchanged (0)
        '''

        # TODO: make sure wage never goes negative ??

        # TODO: consider combining adjust_wages with adjust_workforce

        # make sure wages are not modified in a way that conflicts with
        # the workforce adjustments that will be made during next month
        # !!!TEST REQUIRED!!!

        # deteremine whether wages increase, decrease or remain unchanged for
        # each firm:
        # if (open vacancy last month AND next month)   --> increase wages (+1)
        # if (previous vacancy-free months >= gamma AND
        #     no open vacancy next month)               --> decrease wages (-1)
        # otherwise                                     --> wages unchanged (0)

        # make sure data is consistent - only one of the following can be true:
        # - open vacancy last month: (f.v  > 0)
        # - vacancy-free last month: (f.nv > 0)
        if (np.any((self.v > 0) & (self.nv > 0))):
            n = 0
            while (not (self.v[n] > 0) and (self.nv[n] > 0)):
                n = n + 1
            raise RuntimeError(f"Firm {n} had an open vacancy AND was vacancy-free last month")

        new_v = (self.i < (self.i_phi_lower * self.d)) * 1
        change_type =   ((self.v & new_v) * +1) + \
                        (((self.nv >= self.gamma) & ~new_v) * -1)

        self.w = self.w * (1 + (change_type * np.random.uniform(0, self.delta, self.F)))

    def adjust_workforce(self):
        '''
        See section 2.2 of [Lengnick 2012] for details:

        - First establish upper and lower limits for inventory
        - If current inventory is below lower limit, open a new vacancy
          condition: (f.i < (f.i_phi_lower * f.d))
        - If current inventory is above uper limit, fire a randomly choosen employee
          condition: (f.i > (f.i_phi_upper * f.d))

        Each firm can hire/fire at most one employee per month (see footnote 18)
        '''

        # open vacancies
        # - each firm can have at most 1 vacancy
        self.v = ((self.i < (self.i_phi_lower * self.d)) * 1)

        # increment vacancy-free periods if no vacancy is open
        # !!!TEST REQUIRED!!!
        self.nv = (self.v == 0) * (self.nv + 1)

        # fire employees
        # - each firm can fire at most 1 employee
        # - make sure employment is not less than zero (or one?)
        # TODO: fired employees work for one more month
        self.l = np.clip(self.l - ((self.i > (self.i_phi_upper * self.d)) * 1), 0, None)

    def adjust_prices(self):
        '''
        See section 2.2 of [Lengnick 2012] for details:

        - First establish upper and lower limits for price
          marginal costs: (self.l * self.w) / (self.l * self.t_lambda)
        - Increase price if current inventory is less than inventory lower limit
          condition:
        - Decrease price if current inventory is greater then inventory upper limit
          condition:

        '''

        # TODO: optimize these calculations

        # deteremine whether prices increase, descrease or remain unchanged for each firm:
        # if [inventory > upper limit] --> decrease price (-1)
        # if [inventory < lower limit] --> increase price (+1)
        # otherwise                    --> price unchanged (0)
        change_type =   ((self.i > (self.i_phi_upper * self.d)) * -1) + \
                        ((self.i < (self.i_phi_lower * self.d)) * +1)

        # calculate proposed price change
        price_change = change_type * self.p * np.random.uniform(0, self.nu, self.F)

        # calculate current marginal cost
        # [marginal cost] = [wages paid per worker] / [total output per worker]
        marginal_cost = self.w / self.t_lambda

        # price lower bound
        # - increase  (change_type == 1): max(lb, p) - never lower price
        # - no change (change_type == 0): p
        # - decrease  (change_type = -1): min(lb, p) - never raise price
        p_lower_bound = (change_type  > 0) * np.maximum(self.p_phi_lower * marginal_cost, self.p) + \
                        (change_type == 0) * self.p + \
                        (change_type  < 0) * np.minimum(self.p_phi_lower * marginal_cost, self.p)

        # price upper bound
        # - increase  (change_type == 1): max(ub, p) - never lower price
        # - no change (change_type == 0): p
        # - decrease  (change_type = -1): min(ub, p) - never raise price
        p_upper_bound = (change_type  > 0) * np.maximum(self.p_phi_upper * marginal_cost, self.p) + \
                        (change_type == 0) * self.p + \
                        (change_type  < 0) * np.minimum(self.p_phi_upper * marginal_cost, self.p)

        # calculate proposed new price
        new_p = np.clip(self.p + price_change, a_min=p_lower_bound, a_max=p_upper_bound)

        # change price if price change is accepted
        # - only change price if random # is less than theta
        new_price_accepted = ((np.random.uniform(0, 1, self.F)) <= self.theta)
        self.p = (new_price_accepted * new_p) + (~new_price_accepted * self.p)