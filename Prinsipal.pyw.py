# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:33:01 2020

@author: Brando
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
import Pot_AnalisisDatos
import Pot_Entrenamiento_ANN
import Pot_Prediccion

def Datos_Entrenamiento ():
    Datos_Filtrados = Pot_AnalisisDatos.Analisis_Data ()
    Datos_Entrenamiento_ANN_Keras = Pot_Entrenamiento_ANN.Entrenamiento(Datos_Filtrados)
    
def Predict ():
    Prediccion = Pot_Prediccion.Prediccion_Potencia()

Principal = tk.Tk ()

canvas1 = tk.Canvas(Principal, width = 1366, height = 300)
canvas1.pack()

canvas1.config(bg='alice blue') 

Principal.title("ANN - POTENCIA DE GENERACION")


# Definir ícono para la ventana principal (parte superior izquierda de la ventana principal)
Principal.iconbitmap('PAM_logo.ico') 
# Definir tamaño y posición de la ventana principal 
Principal.geometry('1366x700') #+600+200  
# Configurar para impedir redimensionar de la ventana principal
# Principal.resizable(0,0)
# Se establece el color de fondo de la ventana principal RAIZ  
Principal.config(bg='alice blue') 
## Insercion de la imagen de la empresa
image_petro = tk.PhotoImage(file='PAM_Imagen.png') #Importa imagen
tk.Label(Principal, image = image_petro).place(x=525,y=0)  # Ubicacion de la imagen
#%% IMPORTACION DE RESULTADOS

Data_Resul= pd.read_excel('Prediccion.xlsx', sheet_name='TOTAL', header=0)
Result = Data_Resul.interpolate(method='linear', limit_direction='forward')
Tit_Resul = Result.columns     # Titulo de las columnas; (Tit_Resul[0]) toma el dato en la primera posicion
Potencia_Estim = Result.loc[:, (Tit_Resul[2])].values.reshape(-1,1) # Vector Demanda electrica
Activos = pd.DataFrame(Result.loc[:,(Tit_Resul[1]) ]).values.reshape(-1,1) # Matriz Produccion por bloque
Fecha = pd.DataFrame(Result.loc[:,(Tit_Resul[0]) ]).values.reshape(-1,1) # Matriz Produccion por bloque

## Resultados de Prediccion por Activo
Pot_AU = float(np.floor(Potencia_Estim[0]))
Pot_CY = float(np.floor(Potencia_Estim[1]))
Pot_EY = float(np.floor(Potencia_Estim[2]))
Pot_IN = float(np.floor(Potencia_Estim[3]))
Pot_ITT = float(np.floor(Potencia_Estim[4]))
Pot_LA = float(np.floor(Potencia_Estim[5]))
Pot_LI = float(np.floor(Potencia_Estim[6]))
Pot_OY = float(np.floor(Potencia_Estim[7]))
Pot_PA = float(np.floor(Potencia_Estim[8]))
Pot_SA = float(np.floor(Potencia_Estim[9]))
Pot_SH = float(np.floor(Potencia_Estim[10]))


#%% CREACION DE LA BARRA DE MENU DE LA VENTANA PRINCIPAL
menu_bar = tk.Menu(Principal) 

## Creacion item Configuracion
setup_menu = tk.Menu(menu_bar, tearoff=0)
# Agregar los comondos del item Opciones
setup_menu.add_command(label='ENTRENAMIENTO (Mensual)', command = Datos_Entrenamiento) # Creacion comando definir base de datos historica
setup_menu.add_command(label='PREDICCION', command = Predict) # Creacion comando definir base de datos proyecciones de produccion
#setup_menu.add_command(label='Definir destino de resultados') # Creacion comando definir destino de resultados
## Agrego item Opciones al menu
menu_bar.add_cascade(label='Opciones', menu=setup_menu)

Principal.config(menu=menu_bar) 
#%% Botones
# Boton de Prediccion
ButtonForecasting= tk.Button(Principal, text = ' PREDICCION ',bg="blue",fg='white',
         font=('Arial',10),command=Predict).place(x=30,y=25) 

## Texto en ventana Principal

tk.Label(Principal, text = 'Fecha de Prediccion = ' + str(Fecha[0]), bg="alice blue", font=('Arial',14)).place(x=30,y=75)     
tk.Label(Principal, text = 'Potencia de Generacion Total = ' + str(float(Potencia_Estim[11])) + ' [kW]', bg="alice blue", font=('Arial',14)).place(x=30,y=106)

tk.Label(Principal, text='Potencia de Generacion Parciales: ',font=('Arial',14), bg="alice blue", fg="navy").place(x=30,y=160)

label4 = tk.Label(Principal, text='AUCA     CUYABENO     EDEN YUTURI     INDILLANA       ITT       LAGO AGRIO     LIBERTADOR     OSO YURALPA     PALO AZUL     SACHA     SHUSHUFINDI',
                  font=('Arial',13), bg="alice blue",fg="royal blue")
canvas1.create_window(660, 220, window=label4)

tk.Label(Principal, text = str(Pot_AU)+'       '+str(Pot_CY)+'             '+str(Pot_EY)+'             '+str(Pot_IN)+'      '+str(Pot_ITT)+'        '+str(Pot_LA)+'              '+str(Pot_LI)+'                '+str(Pot_OY)+'              '+str(Pot_PA)+'       '+str(Pot_SA)+'         '+str(Pot_SH),
         bg="alice blue", fg="green", font=('Arial',13)).place(x=50,y=235)


#%% CREACION DE GRAFICAS DE HISTORICO DE PRODUCCION Y POTENCIA
Data_Hist = pd.read_excel('Prod_Activos.xlsx', sheet_name='ACTIVO', header=0 )
HistData = Data_Hist.interpolate(method='linear', limit_direction='forward')
Titulos = HistData.columns     # Titulo de las columnas; (Titulos[0]) toma el dato en la primera posicion
Demand = HistData.loc[:, (Titulos[2])].values.reshape(-1,1)/1000 # Vector Demanda electrica
Production = pd.DataFrame(HistData.loc[:,(Titulos[1]) ])/1000 # Matriz Produccion por bloque
V_Production = Production.values # Matriz Produccion por bloque
Time = HistData.loc[:, (Titulos[0])].values.reshape(-1,1)  # Vector dato de tiempo

#plot 1st scatter 
figure3 = plt.Figure(figsize=(7.05,4))
ax3 = figure3.add_subplot()
ax3.plot(Time, Production, color = 'g', label = 'Fluido')
ax3.legend(loc='upper left')
scatter3 = FigureCanvasTkAgg(figure3, Principal) 
scatter3.get_tk_widget().pack(side=tk.LEFT)
#ax3.set_xticklabels(list(Time),rotation=30)
ax3.set_xlabel('Tiempo')
ax3.set_ylabel('Produccion [MBFPD]')
ax3.set_title('HISTORIAL DE PRODUCCION DE FLUIDO')

#plot 2nd scatter 
figure4 = plt.Figure(figsize=(7.05,4))
ax4 = figure4.add_subplot()
ax4.plot(Time,Demand, color = 'b', label = 'Potencia')
ax4.legend(loc='upper left')
scatter4 = FigureCanvasTkAgg(figure4, Principal) 
scatter4.get_tk_widget().pack(side=tk.RIGHT)
#ax4.set_xticklabels(list(),rotation=30)
ax4.set_xlabel('Tiempo')
ax4.set_ylabel('Potencia [MW]')
ax4.set_title('HISTORIAL DE POTENCIA DE GENERACION')

Principal.mainloop()
