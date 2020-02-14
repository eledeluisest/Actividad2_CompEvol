from utils import algortimo_evolutivo

UMBRAL = 0.0001
SD0 = 10000
DESPLAZAMIENT0 = 10
LR = 0.5
MU = 30
LAMBDA = 200
DIMENSIONES = 10
EJECUCIONES = 20
GENERACIONES = 2000
SALIDA = 0.001
medios = []
tiempos = []
cambios = []
FUNCION = 'SCHWEFEL'
MODO = 'desc_y_padres'
if FUNCION == 'SCHWEFEL':
    LIM_INF = -500
    LIM_SUP = 500
elif FUNCION == 'ESFERA':
    LIM_INF = -1 * DESPLAZAMIENT0 * DESPLAZAMIENT0
    LIM_SUP = DESPLAZAMIENT0 * DESPLAZAMIENT0

with open('res/' + FUNCION + '_' + MODO + '.csv', 'w') as f:
    f.write('ejecucion;n_generaciones;medias;tiempos\n')
for i_ej in range(EJECUCIONES):
    ae = algortimo_evolutivo(SD0, t=LR, modo_desc=MODO)
    ae.define_funcion_optimizar(FUNCION, DESPLAZAMIENT0, LIM_INF, LIM_SUP)
    ae.genera_poblacion(MU, DIMENSIONES)
    for i in range(GENERACIONES):
        ae.recombina(LAMBDA)
        ae.muta(umbral=UMBRAL)
        ae.genera_descencencia()
        ae.fin_paso(verbose=False)
        if SALIDA > ae.__fit__:
            break
    with open('res/' + FUNCION + '_' + MODO + '.csv', 'a+') as f:
        f.write(';'.join([str(i_ej), str(ae.__cambios__), ','.join([str(x) for x in ae.__medias__]),
                          ','.join([str(x) for x in ae.__tiempos__])]) + '\n')
