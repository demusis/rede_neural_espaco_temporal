class No:

    def __init__(self, dadosIniti=0):
        self.result = 0
        self.dadosIniti = dadosIniti
        self.somaResult()

    def setDados(self, dadosIniti):
        self.dadosIniti = dadosIniti

    def getDados(self):
        return self.dadosIniti

    def somaResult(self):
        self.result = self.result + self.dadosIniti
