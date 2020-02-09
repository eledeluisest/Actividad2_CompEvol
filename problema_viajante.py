"""
Vamos a optimizar el problema del viajante con un algoritmo generado desde 0 en python
"""

from utils import algoritmo_genetico

N_ITERACIONES = 500
DIF_SALIR = 0.0005  # calculamos la diferencia entre una generacion y la siguiente y la dividimos entre la media de la ultima
INSTANCIAS = 10
N_POB = 10
K_TORNEO = 5
N_PROMEDIO = 30
SWAP_PROBABILITIES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for sp in SWAP_PROBABILITIES:
    for instancia in range(INSTANCIAS):
        for repeticion in range(N_PROMEDIO):
            iteracion = 0
            dif = DIF_SALIR + 1
            # Inicializamos el algoritmo genetico
            AE_viajante = algoritmo_genetico()
            # Cargamos las instancias
            AE_viajante.carga_instancia('data/viajante_100_50.csv')
            # Elegimos la instancia para poder iterar por ellas
            AE_viajante.elige_instancia(instancia)
            print(sp, instancia, repeticion)
            while iteracion < N_ITERACIONES and dif > DIF_SALIR:

                AE_viajante.codifica_fenotipo_viajante()
                # AE_viajante.__fenotipo__
                AE_viajante.genera_poblacion(n_pob=N_POB)
                # AE_viajante.__poblacion__
                AE_viajante.seleccion_parental(k=K_TORNEO)
                # AE_viajante.genera_descenencia()
                AE_viajante.modelo_generacional(swap_prob=sp)

                # Guardamos los resultados
                AE_viajante.escribe_resultados(instancia, repeticion, iteracion)

                # Actualizamos condicion de salida
                if iteracion != 0:
                    dif = (abs(AE_viajante.__medias__[iteracion] -
                               AE_viajante.__medias__[iteracion - 1]) + abs(AE_viajante.__medias__[iteracion] -
                                                                            AE_viajante.__medias__[iteracion - 2])) / (
                                      2 * AE_viajante.__medias__[iteracion])
                iteracion = iteracion + 1

            with open('data/soluciones.csv', 'a') as f:
                f.write(';'.join([str(sp), str(instancia), str(repeticion)])+';'+ '-'.join(
                    [str(x) for x in AE_viajante.solucion()])+'\n')
