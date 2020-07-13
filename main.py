from time import perf_counter
import numpy as np
import pandas as pd
from ClassRedeNeural import RedeNeural
#from ClassMatrix import Matrix

inicio = perf_counter()

#dados = pd.read_csv('var_comf.csv')
#print(dados.head())
#print(dados.describe().transpose())


dados = pd.DataFrame(100*np.random.rand(100, 5)-100, columns=list(['ta', 'tr', 'vel','rh','pmvg']))

log = open('log.txt', 'w')
log.close()



fim = perf_counter()

print(fim - inicio)
log.close()

