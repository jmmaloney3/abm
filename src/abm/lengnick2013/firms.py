import numpy as np

class Firms:
    # initialize firms
    def __init__(self, F):

        # initialize number of firms
        self.F = F
        
        # initialize model parameters
        # - See [Lengnick 2013] Table 1:
        # firm number of vacancy-free months before reducing wage rate (gamma_f)
        self.gamma = np.full(self.F, 24) # [Lengnick 2013] sets this to 24
        # firm wage % change upper bound (delta_f)
        self.delta = np.full(self.F,0.019) # [Lengnick 2013] sets this to 0.019
        # firm inventory upper bound - percentage of previous demand
        self.i_phi_upper = np.full(self.F, 1)  # [Lengnick 2013] sets this to 1
        # firm inventory lower bound - percentage of previous demand
        self.i_phi_lower = np.full(self.F, 0.25)  # [Lengnick 2013] sets this to 0.25

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
        # firm employees - number of households currently employed by each firm
        self.e = np.ones(self.F) # employees set to 1 at start (?)
        # firm vacancies - current open positions
        self.v = np.ones(self.F) # every firm has a vacancy at start
        # firm number of months without vacancy
        self.nv = np.zeros(self.F) # no months w/o vacancy at start

    def adjust_wages(self):
        '''
        See section 2.2 of [Lengnick 2012] for details:

        - Increase wage if vacancy was not filled during previous month
          condition: (v > 0)
        - Decrease wage if no vacancies for last gamma months
          condition: (nv > g)
        - Randomly select wage change with upper limit equal to delta
          np.random.uniform(0, delta, F)
        - Deteremine direction of wage change (increase or decrease):
          ((v > 0) * 1) + ((nv > g) * -1)
        '''

        self.w = self.w * (1 + ((((self.v > 0) * 1) + ((self.nv > self.gamma) * -1)) * np.random.uniform(0, self.delta, self.F)))

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

        # fire employees
        # - each firm can fire at most 1 employee
        # - make sure employment is not less than zero (or one?)
        self.e = np.clip(self.e - ((self.i > (self.i_phi_upper * self.d)) * 1), 0, None)