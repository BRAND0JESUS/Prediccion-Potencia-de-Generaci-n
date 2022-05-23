# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:12:48 2020

@author: Brando
"""
#%% IMPORTACION DE LIBRERIAS A UTILIZAR
# Importa modulo os (comandos referentes al sistema operativo)
import os 
# Importar modulo pandas para importacion/exportacion de archivos
import pandas as pd
# Importar modulo numpy para uso de matrices
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.externals import joblib



def Entrenamiento (Dat_Filtrados):
    matriz_AUCA_3, matriz_CUYABENO_3, matriz_EDEN_YUTURI_3, matriz_INDILLANA_3, matriz_ITT_3, matriz_LAGO_AGRIO_3, matriz_LIBERTADOR_3, matriz_OSO_YURALPA_3, matriz_PALO_AZUL_3, matriz_SACHA_3, matriz_SHUSHUFINDI_3 = Dat_Filtrados
    
    
    #%% REGRESION RED NEURONAL KERAS

    def ANN_Keras (matz3,name_Act):
    
        V_Production, Demand = matz3
        fil_Prod,col_Prod = V_Production.shape
        Training_Percentage = 0.25 # Especificacion del porcentaje de los datos a tomar en el entrenamiento
        ## El siguiente comando realiza la separacion de los datos pero aleatoriamente
        Prod_train,Prod_test,Dem_train,Dem_test = train_test_split(V_Production, Demand, test_size=Training_Percentage, random_state=42)
        Time_train = np.arange(len(Prod_train))  
        Time_test = np.arange(len(Prod_test))
    
        ## Definicion de la Red Neuronal Artificial
        # Creacion del modelo de la red neuronal con keras mediante una funcion
        
        def create_modelANN():
            model = Sequential()  # Definicion del tipo de red 
            #model.add(Dense(6, input_dim=15, activation="tanh",use_bias=False)) 
            model.add(Dense(20, input_dim = col_Prod, activation="linear",use_bias=False)) # Definicion de la primera capa oculta y capa de entrada
            model.add(Dropout(0.2)) # Mejorar el rendimiento de la ANN (neuronas seleccionadas al azar se ignoran durante el entrenamiento)
            #model.add(Dense(8, activation="tanh")) # Definicion de la segunda capa oculta
            model.add(Dense(1, activation='linear')) # Definicion de la capa de salida
            model.compile(loss='mean_absolute_error',optimizer='Nadam',metrics=["mse"]) # Configuracion de la ANN para el entrenamiento
            model.summary() # Se visualiza en el la consola los parametros de entrenamiento durante cada epoca
            return model
   
        model = create_modelANN()
        #Entrenamiento de la red neuronal
        model.fit(Prod_train,Dem_train,epochs=400,batch_size=100)
        ## Almacenamiento del modelo
        joblib_file = name_Act # Nombre con el que se almacenara la ANN
        joblib.dump(model, joblib_file) # Almancenamiento de la ANN
        return (model,Prod_train,Prod_test,Dem_train,Dem_test,Time_train,Time_test)


    #%% RESULTADOS NUMERICOS Y GRAFIAS DE ENTRENAMIENTO Y PRUEBA
    
    def Error_Plots (matz4):
        model,Prod_train,Prod_test,Dem_train,Dem_test,Time_train,Time_test=matz4
    
        def mean_absolute_percentage_error(y_true, y_pred): 
            MAPE1 = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return MAPE1

        ## Pruebas de la ANN datos de entrenamiento
        predictions_Nor = model.predict(Prod_train) # Prediccion de la demanda electrica con el subconjunto de datos de prueba 
        Prediction = predictions_Nor # Transformacion de las predicciones de valores normalizados a reales
        Y_train = Dem_train # Transformacion de las datos historicos de valores normalizados a reales

        ## Resultados numericos datos de entrenamiento
        MSE_train = mean_squared_error(Y_train, Prediction) # Calculo del MSE
        MAPE_train = mean_absolute_percentage_error(Y_train, Prediction) #Calculo de MAPE
        print('MSE DATOS DE ENTRENAMIENTO:',MSE_train)
        print('MAPE DATOS DE ENTRENAMIENTO:',MAPE_train)

        ## Grafica de comparacion de los datos de entrenamiento
        plt.figure(figsize =(16, 8))
        plt.title('Resultados ANN datos de entrenamiento')
        plt.plot(Time_train,Y_train,c='red',label='Potencia Real')
        plt.plot(Time_train,Prediction,c='blue',label='Potencia Estimada')
        plt.legend(loc='upper left')
        plt.show()

        ## Pruebas de la ANN datos de prueba
        predictions_Nor = model.predict(Prod_test) # Prediccion de la demanda electrica con el subconjunto de datos de prueba 
        Prediction = predictions_Nor # Transformacion de las predicciones de valores normalizados a reales
        Y_test = Dem_test # Transformacion de las datos historicos de valores normalizados a reales
        
        ## Resultados numericos datos de prueba
        MSE_test = mean_squared_error(Y_test, Prediction) # Calculo del MSE
        MAPE_test = mean_absolute_percentage_error(Y_test, Prediction) #Calculo de MAPE
        print('MSE DATOS DE PRUEBA:',MSE_test)
        print('MAPE DATOS DE PRUEBA:',MAPE_test)
        
        ## Grafica de comparacion de los datos de prueba
        plt.figure(figsize =(16, 8))
        plt.title('Resultados ANN datos de prueba')
        plt.plot(Time_test,Y_test,c='red',label='Potencia Real')
        plt.plot(Time_test,Prediction,c='blue',label='Potencia Estimada')
        plt.legend(loc='upper left')
        plt.show()
    
        return (MSE_train,MAPE_test,MSE_train,MAPE_test)

    #%% PRINCIPAL

    Datos = pd.ExcelFile('Prod_Activos.xlsx')
    Name_Sheet = Datos.sheet_names
    Act, AU, CY, EY, IN, ITT, LA, LI, OY, PA, SA, SH, AU_Atip, CY_Atip, EY_Atip, IN_Atip, ITT_Atip, LA_Atip, LI_Atip, OY_Atip, PA_Atip, SA_Atip, SH_Atip = Name_Sheet

    matriz_AUCA_4 = ANN_Keras (matriz_AUCA_3,AU)
    matriz_AUCA_5 = Error_Plots (matriz_AUCA_4)

    matriz_CUYABENO_4 = ANN_Keras (matriz_CUYABENO_3,CY)
    matriz_CUYABENO_5 = Error_Plots (matriz_CUYABENO_4)

    
    matriz_EDEN_YUTURI_4 = ANN_Keras (matriz_EDEN_YUTURI_3,EY)
    matriz_EDEN_YUTURI_5 = Error_Plots (matriz_EDEN_YUTURI_4)
    
    matriz_INDILLANA_4 = ANN_Keras (matriz_INDILLANA_3,IN)
    matriz_INDILLANA_5 = Error_Plots (matriz_INDILLANA_4)
    
    matriz_ITT_4 = ANN_Keras (matriz_ITT_3,ITT)
    matriz_ITT_5 = Error_Plots (matriz_ITT_4)

    matriz_LAGO_AGRIO_4 = ANN_Keras (matriz_LAGO_AGRIO_3,LA)
    matriz_LAGO_AGRIO_5 = Error_Plots (matriz_LAGO_AGRIO_4)
    
    matriz_LIBERTADOR_4 = ANN_Keras (matriz_LIBERTADOR_3,LI)
    matriz_LIBERTADOR_5 = Error_Plots (matriz_LIBERTADOR_4)
    
    matriz_OSO_YURALPA_4 = ANN_Keras (matriz_OSO_YURALPA_3,OY)
    matriz_OSO_YURALPA_5 = Error_Plots (matriz_OSO_YURALPA_4)

    
    matriz_PALO_AZUL_4 = ANN_Keras (matriz_PALO_AZUL_3,PA)
    matriz_PALO_AZUL_5 = Error_Plots (matriz_PALO_AZUL_4)
    
    matriz_SACHA_4 = ANN_Keras (matriz_SACHA_3,SA)
    matriz_SACHA_5 = Error_Plots (matriz_SACHA_4)
    
    matriz_SHUSHUFINDI_4 = ANN_Keras (matriz_SHUSHUFINDI_3,SH)
    matriz_SHUSHUFINDI_5 = Error_Plots (matriz_SHUSHUFINDI_4)
    
    return ()
    