import math
INF = math.inf
from deep_EAT import deepEAT

class EAT(deepEAT):
    def __init__(self, matrix, x, y, numStop, fold):
        'Contructor for EAT prune tree'
        self.fold = fold
        self.NSample = len(matrix)

        # Herency
        deepEAT.__init__(self, matrix, x, y, numStop)
        self.fit_deep_EAT()
        self.td = self.tree
        self.td_tree_alpha_list = self.tree_alpha_list

        #List of tree_alpha_list
        self.TAiv = [[]] * self.fold
        self.BestTivs = [[]] * self.fold
        self.BestTivs[0] = None

        #SE Rule
        self.SE = 0.0

        #Best Tk for now
        self.Tk = self.td_tree_alpha_list[0]

        #Generate Lv (test) and notLv (training)
        self.Lv = [[]] * self.fold
        self.notLv = [[]] * self.fold
        self.generateLv()


    'Destructor'
    def __del__(self):
        try:
            # Remove parameters
            del self.BestTivs
            del self.SE
            del self.TAiv
            del self.leaves
            del self.t
            del self.N
            #del self.matrix
            del self.Lv
            del self.notLv
            del self.Tk
            del self.tree_alpha_list
            del self.td_tree_alpha_list
        except Exception:
            pass

    #Function that build EAT tree
    def fit(self):
        ############ PRUNING - SCORES ###############
        self._scores()

        ############ PRUNING - SERule ###############
        self._SERule()

        ############ PRUNING - Select TK ###############
        self._selectTk()

        # Destructor
        self.__del__()

    # =============================================================================
    # generateLv. Function that generates the training and test data set
    # =============================================================================
    def generateLv(self):
        numRowsFold = math.floor(self.NSample / self.fold)

        for v in range(self.fold):
            # Test
            self.Lv[v] = self.matrix.sample(n=numRowsFold).reset_index(drop=True)
            # Training
            self.notLv[v] = self.matrix.drop(list(self.Lv[v].index)).reset_index(drop=True)

    # =============================================================================
    # Predictor.
    # =============================================================================
    def predict(self, data):
        matrix = data.copy()

        for i in range(len(matrix)):
            pred = self._predictor(self.tree, data.iloc[i])
            for j in range(self.nY):
                matrix.loc[i, "p" + str(j+1)] = pred[j]
        return matrix

    def _predictor(self, tree, register):
        ti = 0  # Root node
        while tree[ti]["SL"] != -1:  # Until we don't reach an end node
            if register.iloc[tree[ti]["xi"]] < tree[ti]["s"]:
                ti = self._posIdNode(tree, tree[ti]["SL"])
            else:
                ti = self._posIdNode(tree, tree[ti]["SR"])
        return tree[ti]["y"]

    ########################## Private methods ####################################

    # =============================================================================
    # treesForRCV. Generation of the trees and tree_alpha_list for each v of folds
    # =============================================================================
    def _treesForRCV(self):
        for v in range(self.fold):
            # TRAINING
            deepEAT.__init__(self, self.notLv[v], self.x, self.y, self.numStop)
            self.fit_deep_EAT()
            self.TAiv[v] = self.tree_alpha_list.copy()

    # =============================================================================
    # RCV. Function that for each v (noLv - Lv) generates a list of subtrees T_iv with
    # their alphas so that, given a profile x (register), it calculates which is
    # the best original subtree (obtained in TreeGenerator from Tmax) and, from it,
    # predicts its 'y' (Predictor function)
    # =============================================================================
    def _RCV(self, alphaIprim):
        # Best sub-trees of each v. This list will have the fold size
        BestTivs = [[]] * self.fold
        Rcv = 0.0

        for v in range(self.fold):
            # Go through the list of trees for each v and select a tree for each v that is <= alphaIprim
            Tiv = []
            TivAux = self.TAiv[v][0]

            for i in range(len(self.TAiv[v]) - 1):
                if self.TAiv[v][i]["alpha"] <= alphaIprim:
                    Tiv = self.TAiv[v][i].copy()
                    TivAux = Tiv
            # If there is no tree we are left with the most complex sub-tree
            if len(Tiv) == 0:
                Tiv = TivAux

            BestTivs[v] = Tiv.copy()

            # TEST
            for register in range(len(self.Lv[v])):
                pred = self._predictor(Tiv["tree"], self.Lv[v].iloc[register])
                for j in range(self.nY):
                    Rcv += (self.Lv[v].iloc[register, self.y[j]] - pred[j]) ** 2
        return Rcv/(self.NSample * self.nY), BestTivs

    def _scores(self):
        # TreesForRCV - training
        self._treesForRCV()

        # tree_alpha_list until -1
        for t in range(len(self.td_tree_alpha_list) - 1):
            # Square root of the sum of alphas
            alphaIprim = (self.td_tree_alpha_list[t]["alpha"] * self.td_tree_alpha_list[t + 1]["alpha"]) ** (1 / 2)

            self.td_tree_alpha_list[t]["score"], BestTivsAux = self._RCV(alphaIprim)

            if (self.Tk["score"] > self.td_tree_alpha_list[t]["score"]) or (self.BestTivs[0] is None):
                Tk = self.td_tree_alpha_list[t]  # Tree_alpha with the lowest score

                for v in range(self.fold):
                    self.BestTivs[v] = BestTivsAux[v]

    def _SERule(self):
        s2 = 0.0

        # SERule (makes again Cross-Validation... only Validation)
        for v in range(self.fold):
            for register in range(len(self.Lv[v])):
                pred = self._predictor(self.BestTivs[v]["tree"], self.Lv[v].iloc[register])
                for j in range(self.nY):
                    dif1 = (self.Lv[v].iloc[register, self.y[j]] - pred[j]) ** 2
                    s2 += (dif1 - self.Tk["score"]) ** 2

        self.SE = (s2 / (self.NSample * self.nY) / (self.NSample * self.nY)) ** (1 / 2)

    def _selectTk(self):
        # Select the definitive tree: the one with the smallest size with a SE clearance score
        margin = self.Tk["score"] + self.SE

        #print("   Tk->score = ", self.Tk["score"])
        #print("   SE        = ", self.SE)
        #print("   margin    = ", margin)

        # Select final tree
        for lst in range(len(self.td_tree_alpha_list)):
            #print(self.td_tree_alpha_list[lst]["score"])
            if (self.td_tree_alpha_list[lst]["score"] <= margin) and (len(self.td_tree_alpha_list[lst]["tree"]) < len(self.Tk["tree"])):
                self.Tk = self.td_tree_alpha_list[lst]

        self.tree = self.Tk["tree"]
        #print("   Tk->score Select = ", self.Tk["score"], ", tree size = ", len(self.Tk["tree"]))
