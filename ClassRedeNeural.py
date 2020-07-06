import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
import tensorflow as tf
import logging

class RedeNeural:

    def __init__(self, dadosIniti,nomeNo):
        self.nomeNo = nomeNo
        self.dadosIniti = dadosIniti.copy()
        self.data_todos = None
        self.x_data = None
        self.y_val = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.ta = None
        self.tr = None
        self.vel = None
        self.rh = None
        self.pmvg_count = 0
        self.pmvg_todos = None
        self.pmvg_est = None
        self.feat_cols = None
        self.input_func = None
        self.model = None
        self.predict_input_func_treino = None
        self.pred_gen_treino = None
        self.predictions = None
        self.final_preds = None
        self.data_treino = None
        self.pred_teste = None
        self.predict_input_func_teste = None
        self.pred_gen_teste = None
        self.final_preds_teste = None
        self.data_teste = None
        self.result_eval = None
        self.padronizacao()
        self.redeNeural()
        self.aplicaAvaliaAmostraTreino()
        self.aplicaModeloAmostraTeste()
        self.aplicaModeloTodosDados()

    def setDados(self, dadosIniti):
        self.dadosIniti = dadosIniti.copy()

    def getDados(self):
        return self.dadosIniti
    
    def getNomeNo(self):
        return self.nomeNo
    
    def getModel(self):
        return self.model

    def getTamanhoXTrain(self):
        return len(self.X_train)

    def getTamanhoYTrain(self):
        return len(self.y_train)

    def getDataTreino(self):
        return self.data_treino

    def getTamanhoXTeste(self):
        return len(self.X_test)

    def getTamanhoYTeste(self):
        return len(self.y_test)
    
    def getPmvgTodos(self):
        return self.pmvg_todos
    
    def getPmvgEst(self):
        return self.pmvg_est
    
    def getPmvgTrain(self):
        return self.pmvg_est_train

    def getDataTeste(self):
        return self.data_teste

    def getDataTodos(self):
        return self.data_todos
    
    def getErroQuadradoTeste(self):
        return np.sqrt(mean_squared_error(self.y_test, self.final_preds_teste))**0.5
    
    def getEvaluateInputFuncTeste(self):
        eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=self.X_test,y=self.y_test,batch_size=10,num_epochs=1,shuffle=False)
        self.result_eval = self.model.evaluate(input_fn=eval_input_func)
        return self.result_eval

    def getMediaAbsolutaErroTreino(self):
        return median_absolute_error(self.data_treino['pmvg_est'], self.data_treino['pmvg'])
    
    def getMediaAbsolutaErroTeste(self):
        return median_absolute_error(self.data_teste['pmvg_est'], self.data_teste['pmvg'])

    def getMediaAbsolutaErroTodos(self):
        return median_absolute_error(self.data_todos['pmvg_est'], self.data_todos['pmvg'])

    def padronizacao(self):
        self.x_data = self.dadosIniti[['ta', 'tr', 'vel', 'rh']]
        self.y_val = self.dadosIniti['pmvg']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_data,self.y_val,test_size=0.3,random_state=101)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X_train)
        self.X_train = pd.DataFrame(data=self.scaler.transform(self.X_train),columns=self.X_train.columns,index=self.X_train.index)
        self.X_test = pd.DataFrame(data=self.scaler.transform(self.X_test),columns=self.X_test.columns,index=self.X_test.index)

    def redeNeural(self):
        logging.getLogger("tensorFlow").setLevel(logging.ERROR)
        self.ta = tf.feature_column.numeric_column('ta')
        self.tr = tf.feature_column.numeric_column('tr')
        self.vel = tf.feature_column.numeric_column('vel')
        self.rh = tf.feature_column.numeric_column('rh')
        self.feat_cols = [self.ta,self.tr,self.vel,self.rh]
        self.input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=self.X_train,y=self.y_train,batch_size=10,num_epochs=1000,shuffle=True)
        self.model = tf.estimator.DNNRegressor(hidden_units=[30,100,100,30],feature_columns=self.feat_cols)
        self.model.train(input_fn=self.input_func,steps=10)
        

    def aplicaAvaliaAmostraTreino(self):
        self.predict_input_func_treino = tf.compat.v1.estimator.inputs.pandas_input_fn(x=self.X_train, batch_size=10, num_epochs=1, shuffle=False)
        self.pred_gen_treino = self.model.predict(self.predict_input_func_treino)
        self.predictions = list(self.pred_gen_treino)
        self.final_preds = []
        for pred in self.predictions:
            self.final_preds.append(pred['predictions'][0])
        self.data_treino = self.X_train
        self.data_treino['pmvg_est'] = self.final_preds
        self.data_treino['pmvg'] = self.y_train
        self.pmvg_est_train = self.final_preds

    def aplicaModeloAmostraTeste(self):
        self.predict_input_func_teste = tf.compat.v1.estimator.inputs.pandas_input_fn(x=self.X_test, batch_size=10, num_epochs=1, shuffle=False)
        self.pred_gen_teste = self.model.predict(self.predict_input_func_teste)
        self.pred_teste = list(self.pred_gen_teste)
        self.final_preds_teste = []
        for pred in self.pred_teste:
            self.final_preds_teste.append(pred['predictions'][0])
        self.data_teste = self.X_test
        self.data_teste['pmvg_est'] = self.final_preds_teste
        self.data_teste['pmvg'] = self.y_test
        self.pmvg_est = self.final_preds_teste

    def aplicaModeloTodosDados(self):
        self.data_todos = pd.concat([self.data_treino, self.data_teste], sort=False)
        self.pmvg_todos = self.data_todos['pmvg_est']
        

    def salvaDadosXLSX(self):
        res = pd.ExcelWriter(self.nomeNo+'.xlsx')
        self.data_todos.to_excel(res,'dados',index=False)
        res.save()
    
    def predicaoElemento(self,elemento):
        predict_input = tf.compat.v1.estimator.inputs.pandas_input_fn(x=elemento, batch_size=10, num_epochs=1, shuffle=False)
        pred_gen = self.model.predict(predict_input)
        predictions_gen = list(pred_gen)
        return predictions_gen

