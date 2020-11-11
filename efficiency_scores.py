import pandas as pd
import numpy as np
from docplex.mp.model import Model


class effScores:
    def __init__(self, matrix, x, y, tree):
        self.matrix = matrix.copy()
        self.N = len(self.matrix)
        self.x = x
        self.y = y

        self.nX = len(self.x)
        self.nY = len(self.y)

        self.forest = tree
        self.treeTk = pd.DataFrame(tree)
        self.atreeTk = pd.DataFrame()
        self.ytreeTk = pd.DataFrame()


    def __del__(self):
        del self.treeTk
        del self.atreeTk
        del self.ytreeTk
        del self.x
        del self.y
        del self.nX
        del self.nY

    # =============================================================================
    # Functions to prepare the data structure
    # =============================================================================
    # Prepare "a" and "y"
    def _prepare_a_y(self):
        a = pd.DataFrame.from_records(self.atreeTk)
        y = pd.DataFrame.from_records(self.ytreeTk)

        # Rename columns
        a_col_name = []
        y_col_name = []

        for c in range(len(a.columns)):
            a_col_name.append("a" + str(c))
        for c in range(len(y.columns)):
            y_col_name.append("y" + str(c))

        a.columns = a_col_name
        y.columns = y_col_name

        self.atreeTk = a.T
        self.ytreeTk = y.T

    def _prepare_col_names(self, num_columns):
        # Rename columns
        col_name = []
        for c in range(num_columns):
            col_name.append("t" + str(c))

        self.atreeTk.columns = col_name
        self.ytreeTk.columns = col_name

    def _cplex_fi(self, x, y):
        self.atreeTk = list(self.treeTk[self.treeTk["SL"] == -1]["a"])
        self.ytreeTk = list(self.treeTk[self.treeTk["SL"] == -1]["y"])

        N = len(self.atreeTk) #N_leaves

        # a and y prepare
        self._prepare_a_y()
        self._prepare_col_names(N)

        # create one model instance, with a name
        m = Model(name='fi_EAT')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        fi = {0: m.continuous_var(name="fi")}

        # Constrain 2.4
        name_lambda = {(i): m.binary_var(name="l_{0}".format(i)) for i in range(N)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(N)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.nX):
            # Constrain 2.1
            m.add_constraint(m.sum(self.atreeTk.iloc[i, j] * name_lambda[j] for j in range(N)) <= x[i])

        for i in range(self.nY):
            # Constrain 2.2
            m.add_constraint(m.sum(self.ytreeTk.iloc[i, j] * name_lambda[j] for j in range(N)) >= fi[0] * y[i])

        # objective
        m.maximize(fi[0])

        # Model Information
        # m.print_information()

        sol = m.solve(agent='local')

        # Solution
        if m.solution == None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol

    def fit(self):
        for i in range(len(self.matrix)):
            self.matrix.loc[i, "ScoreEAT"] = self._cplex_fi(self.matrix.loc[i, self.x].to_list(),\
                                                         self.matrix.loc[i, self.matrix.columns[self.y]].to_list())

        self.__del__()

    # =============================================================================
    # FDH
    # =============================================================================
    # Score FDH
    def _fi_FDH(self, x, y):
        # Prepare matrix
        self.atreeTk = self.matrix[self.x]  #xmatrix
        self.ytreeTk = self.matrix.iloc[:, self.y]  #ymatrix

        N = len(self.atreeTk)

        # xk e yk prepare
        self._prepare_a_y()
        self._prepare_col_names(N)

        # create one model instance, with a name
        m = Model(name='fi_FDH')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        fi = {0: m.continuous_var(name="fi")}

        # Constrain 2.4
        name_lambda = {(i): m.binary_var(name="l_{0}".format(i)) for i in range(N)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(N)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.nX):
            # Constrain 2.1
            m.add_constraint(m.sum(self.atreeTk.iloc[i, j] * name_lambda[j] for j in range(N)) <= x[i])

        for i in range(self.nY):
            # Constrain 2.2
            m.add_constraint(m.sum(self.ytreeTk.iloc[i, j] * name_lambda[j] for j in range(N)) >= fi[0] * y[i])

        # objetive
        m.maximize(fi[0])

        # Model Information
        # m.print_information()
        sol = m.solve(agent='local')

        # Solution
        if m.solution == None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol

    def fit_FDH(self):
        for i in range(len(self.matrix)):
            self.matrix.loc[i, "ScoreFDH"] = self._fi_FDH(self.matrix.loc[i, self.x].to_list(),
                                                          self.matrix.loc[i, self.matrix.columns[self.y]].to_list())
        self.__del__()

    # =============================================================================
    # Teórica (Dios)
    # =============================================================================
    def _fi_Theoric(self, x, y):

        # ---------------------- z = ln(y2, y1) ------------------------------------
        z = np.log(y[1] / y[0])

        # -------------- Pasos 2 y 3 para obtener y1*, y2* -------------------------
        # Ln de x1 y x2
        ln_x1 = np.log(x[0])
        ln_x2 = np.log(x[1])
        # Operaciones para ln_y1_ast
        op1 = -1 + 0.5 * z + 0.25 * (z ** 2) - 1.5 * ln_x1
        op2 = -0.6 * ln_x2 + 0.2 * (ln_x1 ** 2) + 0.05 * (ln_x2 ** 2) - 0.1 * ln_x1 * ln_x2
        op3 = 0.05 * ln_x1 * z - 0.05 * ln_x2 * z
        ln_y1_ast = -(op1 + op2 + op3)

        # Y de ese valor determinamos y1*=exp(ln(y1*))
        y1_ast = np.exp(ln_y1_ast)
        # P3(Calculamos ln(y2*) como z + ln(y1*). Del ln(y2*), sacamos y2* = exp(ln(y2*))
        # y2_ast = np.exp(ln_y1_ast + z)

        # ------------------ Obtener fi --------------------------------------------
        fi_y1 = y1_ast / y[0]
        # fi_y2 = y2_ast / y[1]

        return fi_y1

    def fit_Theoric(self):
        for i in range(len(self.matrix)):
            print(self.matrix.loc[i, self.x].to_list())
            self.matrix.loc[i, "ScoreTheoric"] = self._fi_Theoric(self.matrix.loc[i, self.x].to_list(),
                                                                  self.matrix.loc[i, self.matrix.columns[self.y]].to_list())
        self.__del__()