import pandas as pd
import math
INF = math.inf
from docplex.mp.model import Model

class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

class EXIT(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return style.YELLOW + "\n\n" + self.message + style.RESET


class Scores:
    def __init__(self, matrix, x, y, tree):
        self.matrix = matrix.loc[:, x + y]  # Order variables
        self.N = len(self.matrix)
        self._check_columnsX_in_data(matrix, x)

        self.x = matrix.columns.get_indexer(x).tolist()  # Index var.ind in matrix
        self.y = matrix.columns.get_indexer(y).tolist()  # Index var. obj in matrix
        self.nX = len(self.x)
        self.nY = len(self.y)

        self.tree = tree

        self.atreeTk = pd.DataFrame()
        self.ytreeTk = pd.DataFrame()
        self.N_leaves = 0

    def __del__(self):
        del self.nX
        del self.nY
        del self.tree
        del self.atreeTk
        del self.ytreeTk
        del self.N_leaves

    def _check_columnsX_in_data(self, matrix, x):
        cols = x
        for col in cols:
            if col not in matrix.columns.tolist():
                raise EXIT("ERROR. The names of the inputs are not in the dataset")

    def _prepareData(self, matrix, x, y):

        pass
    
    def _prepare_a(self, a_y_treeTk, name):
        a = pd.DataFrame.from_records(a_y_treeTk)
        # Rename columns
        col_name = []
        for c in range(len(a.columns)):
            col_name.append(name + str(c))
        a.columns = col_name
        return a.T

    def _prepare_col_names(self, df, columns, name):
        # Rename columns
        col_name = []
        for c in range(columns):
            col_name.append(name + str(c))

        df.columns = col_name
        return df

    def _prepare_model(self):
        # Prepare tree
        treeTk = pd.DataFrame(self.tree)
        self.atreeTk = list(treeTk[treeTk["SL"] == -1]["a"])
        self.ytreeTk = list(treeTk[treeTk["SL"] == -1]["y"])

        self.N_leaves = len(self.atreeTk)

        # a prepare
        self.atreeTk = self._prepare_col_names(self._prepare_a(self.atreeTk, "a"), self.N_leaves, "t")
        # y prepare
        self.ytreeTk = self._prepare_col_names(self._prepare_a(self.ytreeTk, "y"), self.N_leaves, "t")

    def _prepare_model_DEA_FDH(self):
        # a prepare
        self.atreeTk = self._prepare_col_names(self._prepare_a(self.atreeTk, "a"), self.N, "t")
        # y prepare
        self.ytreeTk = self._prepare_col_names(self._prepare_a(self.ytreeTk, "y"), self.N, "t")


    def _scoreEAT_BCC_output(self, xn, yn):
        self._prepare_model()

        # create one model instance, with a name
        m = Model(name='fi_EAT')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        fi = {0: m.continuous_var(name="fi")}

        # Constrain 2.4
        name_lambda = {i: m.binary_var(name="l_{0}".format(i)) for i in range(self.N_leaves)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N_leaves)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.nX):
            # Constrain 2.1
            m.add_constraint(m.sum(self.atreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N_leaves)) <= xn[i])

        for i in range(self.nY):
            # Constrain 2.2
            m.add_constraint(m.sum(self.ytreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N_leaves)) >= fi[0] * yn[i])

        # objetive
        m.maximize(fi[0])

        m.solve(agent='local')
        # Solution
        if m.solution is None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol

    def _scoreDEAEAT_BCC_output(self, x, y):
        self._prepare_model()

        # create one model instance, with a name
        m = Model(name='fi_EAT')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        fi = {0: m.continuous_var(name="fi")}

        # Constrain 2.4
        name_lambda = {i: m.continuous_var(name="l_{0}".format(i)) for i in range(self.N_leaves)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N_leaves)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.nX):
            # Constrain 2.1
            m.add_constraint(m.sum(self.atreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N_leaves)) <= x[i])

        for i in range(self.nY):
            # Constrain 2.2
            m.add_constraint(m.sum(self.ytreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N_leaves)) >= fi[0] * y[i])

        # objetive
        m.maximize(fi[0])

        # Model Information
        # m.print_information()

        sol = m.solve(agent='local')

        # Solución
        if (m.solution == None):
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol

    def _scoreEAT_BCC_input(self, xn, yn):
        self._prepare_model()

        # create one model instance, with a name
        m = Model(name='fi_EAT')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        fi = {0: m.continuous_var(name="fi")}

        # Constrain 2.4
        name_lambda = {i: m.binary_var(name="l_{0}".format(i)) for i in range(self.N_leaves)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N_leaves)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.nX):
            # Constrain 2.1
            m.add_constraint(m.sum(self.atreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N_leaves)) <= fi[0] * xn[i])

        for i in range(self.nY):
            # Constrain 2.2
            m.add_constraint(m.sum(self.ytreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N_leaves)) >= yn[i])

        # objetive
        m.minimize(fi[0])

        m.solve(agent='local')
        if m.solution is None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol

    def _scoreEAT_DDF(self, xn, yn):
        self._prepare_model()
        # create one model instance, with a name
        m = Model(name='beta_EAT')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        beta = {0: m.continuous_var(name="beta")}

        # Constrain 2.4
        name_lambda = {i: m.binary_var(name="l_{0}".format(i)) for i in range(self.N_leaves)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N_leaves)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.nX):
            # Constrain 2.1
            m.add_constraint(
                m.sum(self.atreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N_leaves)) <= xn[i] - beta[0] * xn[i])

        for i in range(self.nY):
            # Constrain 2.2
            m.add_constraint(
                m.sum(self.ytreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N_leaves)) >= yn[i] + beta[0] * yn[i])

        # objetive
        m.maximize(beta[0])

        m.solve(agent='local')
        if m.solution is None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol


    def BCC_output_EAT(self):
        nameCol = "BCC_output_EAT"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            self.matrix.loc[i, nameCol] = self._scoreEAT_BCC_output(self.matrix.iloc[i, self.x].to_list(),
                                                                    self.matrix.iloc[i, self.y].to_list())

    def BCC_output_CEAT(self):
        nameCol = "BCC_output_CEAT"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            # fi_EAT(X = ["x1", "x2"], Y = ["y1", "y2"], tree)
            self.matrix.loc[i, nameCol] = self._scoreDEAEAT_BCC_output(self.matrix.iloc[i, self.x].to_list(),
                                                            self.matrix.iloc[i, self.y].to_list())

    def BCC_input_EAT(self):
        nameCol = "BCC_input_EAT"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            self.matrix.loc[i, nameCol] = self._scoreEAT_BCC_input(self.matrix.iloc[i, self.x].to_list(),
                                                                   self.matrix.iloc[i, self.y].to_list())

    def DDF_EAT(self):
        nameCol = "DDF_EAT"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            self.matrix.loc[i, nameCol] = self._scoreEAT_DDF(self.matrix.iloc[i, self.x].to_list(),
                                                             self.matrix.iloc[i, self.y].to_list())

    #FDH
    def _scoreFDH_BCC_output(self, x, y):
        # Prepare matrix
        self.atreeTk = self.matrix.iloc[:, self.x]  # xmatrix
        self.ytreeTk = self.matrix.iloc[:, self.y]  # ymatrix

        self._prepare_model_DEA_FDH()

        # create one model instance, with a name
        m = Model(name='fi_FDH')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        fi = {0: m.continuous_var(name="fi")}

        # Constrain 2.4
        name_lambda = {i: m.binary_var(name="l_{0}".format(i)) for i in range(self.N)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.nX):
            # Constrain 2.1
            m.add_constraint(m.sum(self.atreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N)) <= x[i])

        for i in range(self.nY):
            # Constrain 2.2
            m.add_constraint(m.sum(self.ytreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N)) >= fi[0] * y[i])

        # objetive
        m.maximize(fi[0])

        # Model Information
        # m.print_information()

        m.solve()

        # Solución
        if m.solution == None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol

    def _score_DDF_FDH(self, x, y):
        # Prepare matrix
        self.atreeTk = self.matrix.iloc[:, self.x]  # xmatrix
        self.ytreeTk = self.matrix.iloc[:, self.y]  # ymatrix

        self._prepare_model_DEA_FDH()

        # create one model instance, with a name
        m = Model(name='beta_FDH')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        beta = {0: m.continuous_var(name="beta")}

        # Constrain 2.4
        name_lambda = {i: m.binary_var(name="l_{0}".format(i)) for i in range(self.N)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.nX):
            # Constrain 2.1
            m.add_constraint(m.sum(self.atreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N)) <= x[i] - beta[0] * x[i])

        for i in range(self.nY):
            # Constrain 2.2
            m.add_constraint(m.sum(self.ytreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N)) >= y[i] + beta[0] * y[i])

        # objetive
        m.maximize(beta[0])

        # Model Information
        # m.print_information()

        m.solve()

        # Solución
        if m.solution == None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol

    def BCC_output_FDH(self):
        nameCol = "BCC_output_FDH"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            self.matrix.loc[i, nameCol] = self._scoreFDH_BCC_output(self.matrix.iloc[i, self.x].to_list(),
                                                         self.matrix.iloc[i, self.y].to_list())

    def DDF_FDH(self):
        nameCol = "DDF_FDH"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            self.matrix.loc[i, nameCol] = self._score_DDF_FDH(self.matrix.iloc[i, self.x].to_list(),
                                                   self.matrix.iloc[i, self.y].to_list())

    #DEA
    def _scoreDEA_BCC_output(self, x, y):
        # Prepare matrix
        self.atreeTk = self.matrix.iloc[:, self.x]  # xmatrix
        self.ytreeTk = self.matrix.iloc[:, self.y]  # ymatrix

        self._prepare_model_DEA_FDH()

        # create one model instance, with a name
        m = Model(name='beta_DEA')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        beta = {0: m.continuous_var(name="beta")}

        # Constrain 2.4
        name_lambda = {i: m.continuous_var(name="l_{0}".format(i)) for i in range(self.N)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.nX):
            # Constrain 2.1
            m.add_constraint(m.sum(self.atreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N)) <= x[i])

        for i in range(self.nY):
            # Constrain 2.2
            m.add_constraint(m.sum(self.ytreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N)) >= beta[0] * y[i])

        # objetive
        m.maximize(beta[0])

        # Model Information
        # m.print_information()

        m.solve()

        # Solución
        if m.solution == None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol

    def _score_DDF_DEA(self, x, y):
        # Prepare matrix
        self.atreeTk = self.matrix.iloc[:, self.x]  # xmatrix
        self.ytreeTk = self.matrix.iloc[:, self.y]  # ymatrix

        self._prepare_model_DEA_FDH()

        # create one model instance, with a name
        m = Model(name='beta_DEA_DDF')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        beta = {0: m.continuous_var(name="beta")}

        # Constrain 2.4
        name_lambda = {i: m.continuous_var(name="l_{0}".format(i)) for i in range(self.N)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N)) == 1)  # sum(lambda) == 1

        # Constrain 2.1 y 2.2
        for i in range(self.nX):
            cons = m.sum(self.atreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N)) + beta[0]*x[i] <= x[i]
            # Constrain 2.1
            m.add_constraint(cons)

        for i in range(self.nY):
            cons = m.sum(self.ytreeTk.iloc[i, j] * name_lambda[j] for j in range(self.N)) - beta[0]*y[i] >= y[i]
            # Constrain 2.2
            m.add_constraint(cons)
            #print(cons)

        # objetive
        m.maximize(beta[0])

        # Model Information
        #m.print_information()
        m.solve()

        # Solución
        if m.solution == None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol

    def BCC_output_DEA(self):
        nameCol = "BCC_output_DEA"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            self.matrix.loc[i, nameCol] = self._scoreDEA_BCC_output(self.matrix.iloc[i, self.x].to_list(),
                                                                    self.matrix.iloc[i, self.y].to_list())

    def DDF_DEA(self):
        nameCol = "DDF_DEA"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            self.matrix.loc[i, nameCol] = self._score_DDF_DEA(self.matrix.iloc[i, self.x].to_list(),
                                                   self.matrix.iloc[i, self.y].to_list())
