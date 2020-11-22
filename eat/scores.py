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
        self.matrix = matrix
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

    def _scoreEAT_BBC_output(self, xn, yn):
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

    def _scoreEAT_BBC_input(self, xn, yn):
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
        m.maximize(fi[0])

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

    def BBC_output_EAT(self):
        nameCol = "BBC_output_EAT"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            self.matrix.loc[i, nameCol] = self._scoreEAT_BBC_output(self.matrix.loc[i, self.x].to_list(),
                                                                    self.matrix.loc[i, self.matrix.columns[self.y]].to_list())

    def BBC_input_EAT(self):
        nameCol = "BBC_input_EAT"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            self.matrix.loc[i, nameCol] = self._scoreEAT_BBC_input(self.matrix.loc[i, self.x].to_list(),
                                                                   self.matrix.loc[i, self.matrix.columns[self.y]].to_list())

    def DDF_EAT(self):
        nameCol = "DDF_EAT"
        self.matrix.loc[:, nameCol] = 0

        for i in range(len(self.matrix)):
            self.matrix.loc[i, nameCol] = self._scoreEAT_DDF(self.matrix.loc[i, self.x].to_list(),
                                                             self.matrix.loc[i, self.matrix.columns[self.y]].to_list())
