import numpy as np
import pandas as pd
import time

class Data2:
    def __init__(self, sd, N, nX):
        # Seed random
        np.random.seed(sd)

        self.N = N
        self.nX = nX

        self.data = pd.DataFrame()

        # Generate nX
        for x in range(self.nX):
            self.data["x" + str(x + 1)] = np.random.uniform(1, 10, self.N)

        u = abs(np.random.normal(0, 0.4, self.N))

        if self.nX == 3:
            y = 3 + (self.data["x1"] ** 0.05) + (self.data["x2"] ** 0.15) + (self.data["x3"] ** 0.3)
            self.data["y"] = y - u
            self.data["yD"] = y

        elif self.nX == 6:
            y = 3 + (self.data["x1"] ** 0.05) + (self.data["x2"] ** 0.001) + (self.data["x3"] ** 0.004) \
                + (self.data["x4"] ** 0.045) + (self.data["x5"] ** 0.1) + (self.data["x6"] ** 0.3)
            self.data["y"] = y - u
            self.data["yD"] = y

        elif self.nX == 9:
            y = 3 + (self.data["x1"] ** 0.005) + (self.data["x2"] ** 0.001) + (self.data["x3"] ** 0.004) \
                + (self.data["x4"] ** 0.005) + (self.data["x5"] ** 0.001) + (self.data["x6"] ** 0.004) \
                + (self.data["x7"] ** 0.08) + (self.data["x8"] ** 0.1) + (self.data["x9"] ** 0.3)
            self.data["y"] = y - u
            self.data["yD"] = y

        elif self.nX == 12:
            y = 3 + (self.data["x1"] ** 0.005) + (self.data["x2"] ** 0.001) + (self.data["x3"] ** 0.004) \
                + (self.data["x4"] ** 0.005) + (self.data["x5"] ** 0.001) + (self.data["x6"] ** 0.004) \
                + (self.data["x7"] ** 0.08) + (self.data["x8"] ** 0.05) + (self.data["x9"] ** 0.05) \
                + (self.data["x10"] ** 0.075) + (self.data["x11"] ** 0.025) + (self.data["x12"] ** 0.2)
            self.data["y"] = y - u
            self.data["yD"] = y

        elif self.nX == 15:
            y = 3 + (self.data["x1"] ** 0.005) + (self.data["x2"] ** 0.001) + (self.data["x3"] ** 0.004) \
                + (self.data["x4"] ** 0.005) + (self.data["x5"] ** 0.001) + (self.data["x6"] ** 0.004) \
                + (self.data["x7"] ** 0.08) + (self.data["x8"] ** 0.05) + (self.data["x9"] ** 0.05) \
                + (self.data["x10"] ** 0.05) + (self.data["x11"] ** 0.025) + (self.data["x12"] ** 0.025) \
                + (self.data["x13"] ** 0.025) + (self.data["x14"] ** 0.025) + (self.data["x10"] ** 0.15)
            self.data["y"] = y - u
            self.data["yD"] = y
        else:
            print("Error. Input size")
