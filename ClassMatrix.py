from ClassNo import No


class Matrix:

    def __init__(self, tamanho):
        self.length = 0
        self.matrix = []
        self.row = []
        self.tamanho = tamanho

    def setNo(self, no: No):
        if self.length % self.tamanho == 0:
            self.row.append(no)
            self.matrix.append(self.row)
            self.row = []
            self.length += 1
        else:
            self.row.append(no)

    def getNo(self, i:int, j:int):
        if i == self.length:
            if j <= self.tamanho:
                return self.row[j]
        else:
            if i < self.length:
                if j <= self.tamanho:
                    return self.matrix[i][j]
        return False

    def finalizaMatrix(self):
        self.matrix.append(self.row)
