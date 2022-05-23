# Predicción de Potencia de Generación
Red Neuronal para la predicción de la potencia de generación de los campos operados por la empresa petrolera estatal Ecuatoriana "PETROAMAZONAS" los cuales son los siguientes: 
AUCA
CUYABENO
EDEN YUTURI
INDILLANA
ITT
LAGO AGRIO
LIBERTADOR
OSO YURALPA
PALO AZUL
SACHA
SHUSHUFINDI

El presente proyecto se encuentra dividido en secciones donde:
- Principal.pyw.py -> corresponde GUI (graphical user interface) para interactuar con el usuario.
- Pot_AnalisisDatos.py -> corresponde al algoritmo para limpieza ay filtracion de datos de la producción de petróleo de los campos 11 campos operados por la petrolera estatal.
- Pot_Entrenamiento_ANN.py -> corresponde al algoritmo de entrenamiento con el uso de la librería de sklearn, ademas permite guardar el entrenamiento mediante joblib para cada campo.
- Pot_Prediccion.py -> algoritmo que permite predecir la Potencia de Generación con un porcentaje de error de entre 0.4 % y 1.02% 
