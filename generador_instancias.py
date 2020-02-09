"""
08/12/2019
Luis Esteban

Generador de instancias para el problema del viajante

"""

from utils import generador_ejemplos

# Este codigo genera ejemplos del problema del viajante de forma aleatoria uniforme

puntos = 100
limites = [10, 10]
instancias = 50
viajante = generador_ejemplos(problema='VIAJANTE', data_path='data/ejemplo1.csv')
viajante.viajante_ini(n=puntos, lista_lim=limites)
viajante.genera_instancias(instancias)
