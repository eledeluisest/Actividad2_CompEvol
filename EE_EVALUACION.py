
def eva(funcion, modo1, modo2):
    import pandas as pd
    import matplotlib.pyplot as plt
    UMBRAL = 0.0001
    SD0 = 10000
    DESPLAZAMIENT0 = 10
    LR = 0.5
    LR2 = 0.5
    MU = 30
    LAMBDA = 200
    DIMENSIONES = 10
    EJECUCIONES = 20
    GENERACIONES = 2000
    SALIDA = 0.001
    medios = []
    tiempos = []
    cambios = []
    # FNNCIONS POSIBLES : 'ESFERA' SCHWEFEL
    FUNCION = funcion
    # MODOS POSIBLES : 'solo_desc' 'desc_y_padres'
    MODO = modo1
    # MODOS MUTA POSIBLES : 'un_paso' 'n_pasos'
    MODO_MUTA = modo2
    if FUNCION == 'SCHWEFEL':
        LIM_INF = -500
        LIM_SUP = 500
    elif FUNCION == 'ESFERA':
        LIM_INF = -1 * DESPLAZAMIENT0 * DESPLAZAMIENT0
        LIM_SUP = DESPLAZAMIENT0 * DESPLAZAMIENT0
    file_name = '_'.join([str(x) for x in
                          [FUNCION, MODO, MODO_MUTA, 'LR', LR, 'LR2', LR2, 'SD0', SD0, 'D', DIMENSIONES, 'M', MU, 'L',
                           LAMBDA]]) + '.csv'


    df_res = pd.read_csv('res/'+file_name,sep=';')
    # Procesamos los datos
    df_res['medias_list'] = df_res.medias.apply(lambda x: [float(y) for y in x.split(',')])
    df_res['tiempos_list'] = df_res.tiempos.apply(lambda x: [float(y) for y in x.split(',')])
    df_res['fin_fit'] = df_res.medias_list.apply(lambda x: x[-1] )
    # Calculamos las métricas y las escribimos por pantallas
    SR = sum((df_res['fin_fit']) < UMBRAL * 10)/ len(df_res)
    MBF = df_res['fin_fit'].mean()
    AES = df_res.loc[df_res['fin_fit'] < UMBRAL * 10,'medias_list'].apply(lambda x: len(x)).mean()
    S_medias = pd.Series(df_res['medias_list'][0])
    S_tiempos = pd.Series(df_res['tiempos_list'][0])
    print([MODO, MODO_MUTA])
    print(str(SR).replace('.',','))
    print(str(MBF).replace('.',','))
    print(str(AES).replace('.',','))

    # Pintamos las gráficas de convergencia.
    fig = plt.figure()
    plt.title(FUNCION+'_'+MODO+'_'+MODO_MUTA)
    for i in range(len(df_res)):
        pd.Series(df_res['medias_list'][i]).plot()
    plt.ylabel('fitness')
    plt.xlabel('Número de generaciones')
    fig.show()



    # MODOS POSIBLES : 'solo_desc' 'desc_y_padres'
    # MODOS MUTA POSIBLES : 'un_paso' 'n_pasos'
eva('SCHWEFEL','solo_desc','un_paso')

