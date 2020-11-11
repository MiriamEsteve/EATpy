import numpy as np
import math
INF = math.inf


class predictions:
    def __init__(self, matrix, x, y, tree):
        self.matrix = matrix.copy()
        self.N = len(self.matrix)
        self.x = x
        self.y = y

        self.nX = len(self.x)
        self.nY = len(self.y)

        self.tree = tree

    def _posIdNode(self, tree, idNode):
        for i in range(len(tree)):
            if tree[i]["id"] == idNode:
                return i
        return -1

    def _predictor(self, tree, register):
        ti = 0  # Root node
        while tree[ti]["SL"] != -1:  # Until we don't reach an end node
            if register.iloc[tree[ti]["xi"]] < tree[ti]["s"]:
                ti = self._posIdNode(tree, tree[ti]["SL"])
            else:
                ti = self._posIdNode(tree, tree[ti]["SR"])
        return tree[ti]["y"]

    def yEAT(self):
        for i in range(self.N):
            self.matrix.loc[i, "yEAT"] = self._predictor(self.tree, self.matrix.iloc[i])[0]

    # =============================================================================
    # FDH
    # =============================================================================
    def _FDH(self, XArray):
        yMax = -INF
        for n in range(self.N):
            newMax = True
            for i in range(len(XArray)):
                if i < self.y[0]:
                    if self.matrix.iloc[n, i] > XArray[i]:
                        newMax = False
                        break
                # Else if en caso de que la 'y' no esté en la última columna
                elif i > self.y[0]:
                    if self.matrix.iloc[n, i + 1] > XArray[i]:
                        newMax = False
                        break

            if newMax and yMax < self.matrix.iloc[n, self.y[0]]:
                yMax = self.matrix.iloc[n, self.y[0]]

        return yMax

    # yFDH(dataset, PosiciónVariableConsecuente)
    def yFDH(self):
        for i in range(self.N):
            self.matrix.loc[i, "yFDH"] = self._FDH(self.matrix.loc[i, self.x])
