import numpy as np

class Firms:
    # initialize firms
    def __init__(self, F):

        # initialize number of firms
        self.F = F
        
        # initialize model parameters
        # firm number of vacancy-free months before reducing wage rate (gamma_f)
        self.gamma = np.full(F, 24) # [Lengnick 2013] sets this to 24
        # firm wage % change upper bound (delta_f)
        self.delta = np.full(F,0.019) # [Lengnick 2013] sets this to 0.019

        # initial conditions (TBD)
        # firm liquidity (m_f) - current "bank account" balance
        self.m = np.zeros(F) # bank balance is zero at start (?)
        # firm inventory (i_f) - current inventory levels
        self.i = np.full(F, 5) # inventory set to 5 at start (?)
        # firm wage (w_f) - current wage paid to employees
        self.w = np.ones(F) # wage set to 1 at start (?)
        # firm price (p_f) - current price
        self.p = np.ones(F) # price set to 1 at start (?)
        # firm vacancies - current open positions
        self.v = np.ones(F) # every firm has a vacancy at start
        # firm number of months without vacancy
        self.nv = np.zeros(F) # no months w/o vacancy at start

    def adjust_wages(self):
        '''
        See section 2.2 of [Lengnick 2012] for details:

        - Increase wage if vacancy was not filled during prevous month
          condition: (v > 0)
        - Decrease wage if no vacancies for last gamma months
          condition: (nv > g)
        - Randomly select wage change with upper limit equal to delta
          np.random.uniform(0, delta, F)
        - Deteremine direction of wage change:
          ((v > 0) * 1) + ((nv > g) * -1)
        '''

        self.w = self.w * (1 + ((((self.v > 0) * 1) + ((self.nv > self.gamma) * -1)) * np.random.uniform(0, self.delta, self.F)))

