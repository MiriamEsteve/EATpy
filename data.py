import numpy as np
import pandas as pd

#sd = semilla, N = tam.muestra, nX = num. X, nY = num. Y, border = % frontera, noise = ruido {0/1}
class Data:
    def __init__(self, sd, N):
        self.sd = sd
        self.N = N
        self.nX = 2
        self.nY = 2

        # Seed random
        np.random.seed(self.sd)

        # DataFrame vacio
        self.data = pd.DataFrame()

        # P1 (Generar de forma aleatoria x1, x2 y z)
        self._generate_X_Z()



    def _generate_X_Z(self):
        # Generar nX
        for x in range(self.nX):
            # Generar X's
            self.data["x" + str(x + 1)] = np.random.uniform(5, 50, self.N)

        # Generar z
        z = np.random.uniform(-1.5, 1.5, self.N)

        # Generar cabeceras nY
        for y in range(self.nY):
            self.data["y" + str(y + 1)] = None

        # Ln de x1 y x2
        ln_x1 = np.log(self.data["x1"])
        ln_x2 = np.log(self.data["x2"])

        # Operaciones para ln_y1_ast
        op1 = -1 + 0.5 * z + 0.25 * (z ** 2) - 1.5 * ln_x1
        op2 = -0.6 * ln_x2 + 0.2 * (ln_x1 ** 2) + 0.05 * (ln_x2 ** 2) - 0.1 * ln_x1 * ln_x2
        op3 = 0.05 * ln_x1 * z - 0.05 * ln_x2 * z
        ln_y1_ast = -(op1 + op2 + op3)

        # Y de ese valor determinamos y1*=exp(ln(y1*))
        self.data["y1"] = np.exp(ln_y1_ast)

        # P3(Calculamos ln(y2*) como z + ln(y1*). Del ln(y2*), sacamos y2* = exp(ln(y2*))
        self.data["y2"] = np.exp(ln_y1_ast + z)
