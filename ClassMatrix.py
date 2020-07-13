from ClassRedeNeural import RedeNeural
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
from pyKriging.testfunctions import testfunctions
import pandas as pd


class Matrix:

    def __init__(self, colunas, redeneural):
        self.matrix = pd.DataFrame(columns = colunas)
        self.colunas = colunas
        self.redeneural = redeneural
        self.kikage = None
        self.X = None
        self.y = None
        

    def setLinhaMatrix(self, item):
        self.matrix.append(item)
        
    def setNovaColunaMatrix(self, item):
        self.matrix[item] = []

    def getShapeMatrix(self):
        return self.matrix.shape

    def getItemMatrix(self, i:int):
        return self.matrix[i]
    
    def getTotalElementosLinhaMatrix(self, i:int):
        return self.matrix[i].count
    
    def getColunasMatrix(self):
        return self.matrix.columns
    
    def getMatrix(self):
        return self.matrix
    
    def krigagemMatrix(self, colY,colX,colZ):
        
        self.X = list(self.matrix[colY, colX,colZ])
        testfun = testfunctions().squared
        self.y = testfun(self.X)
        self.k = kriging(self.X, self.y, testfunction=testfun, testPoints=300)
        
        self.k.train()
        self.k.snapshot()
        
        
    def predicaoElementoKrigagem(self,elemento, i:int):
        pred_gen = self.k.predict(elemento)
        if 'krigagem' in self.matrix.columns:
            self.matrix[i]['krigagem'] = pred_gen
        else:
            self.setNovaColunaMatrix('krigagem')
            
        return predictions_gen
        
        