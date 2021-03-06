"""
Generamos clases para cada una de las fases del EA

Dependencias:
import time
import random
import os
import numpy as np
import warnings

"""
import time
import random
import os
import numpy as np
import warnings
import pandas as pd


class generador_ejemplos:
    """
    Esta clase genera ejmplos de problemas para la optimización con Algoritmos genéticos u otros
    """

    def __init__(self, problema, seed_gen=None, seed_ea=None, data_path=None):
        """
        Constructor
        :param problema: Clave del problema
        :param seed_gen: Número para fijar la generación aleatoria
        :param seed_ea: Número para fijar la generación aleatoria
        :param data_path: ruta dónde se almacenará la instancia
        """
        self.__problema__ = problema
        self.__seed_gen__ = seed_gen
        self.__seed_ea__ = seed_ea

        print('Se generaran instancias del problema: ' + problema)
        # Comprobamos la existencia del directorio, sino lo creamos
        if type(data_path) is type(None):
            self.__problema_path__ = 'data/CE_instancia_' + problema + '_' + str(time.time()) + '.csv'
            if not os.path.exists('data/'):
                os.makedirs('data/')
        else:
            self.__problema_path__ = data_path
        # Informamos sobre los métodos a utilizar
        if problema == 'VIAJANTE':
            print(
                'Utilizar metodos viajante_ini para fijar los parametros y genera instancia para generar las instancias.')

    def __viajante_gen__(self):
        """
        Genración de puntos del problema del viajante
        :return: Línea con instancia del problema en formato string
        """
        puntos = []
        p = []
        if type(self.__seed_gen__) is not type(None):
            random.seed = self.__seed_gen__
        # Iteramos por los puntos existentes y los añadimos a la lista que después escribiremos.
        # Permitimmos que dos puntos estén en la misma posición
        for n in range(self.__NPuntos__):
            for limite in self.__limites__:
                p.append(str(random.uniform(0, limite)))
            puntos.append(p)
            p = []
        # Devolvemos la cadena para su escritura
        return ';'.join([','.join(x) for x in puntos]) + '\n'

    def viajante_ini(self, n, lista_lim):
        """
        Método de inicialización de parámetros
        :param n: número de puntos
        :param lista_lim: lista con las dimensiones del hiperplano dónde generar los puntos
        :return:
        """
        self.__NPuntos__ = n
        self.__dimension__ = len(lista_lim)
        self.__limites__ = lista_lim

    def genera_instancias(self, n, mode='file'):
        """
        Genera y graba las instancias
        :param n: Número de instancias
        :param mode: Modo de guardado. Solo implementado a fichero.
        :return:
        """
        if mode == 'file':
            with open(self.__problema_path__, 'w+') as f:
                for i in range(n):
                    if self.__problema__ == 'VIAJANTE':
                        # print(self.__viajante_gen__())
                        f.write(self.__viajante_gen__())


class algoritmo_genetico():
    """
    Esta clase contiene lo necesario para ejecutar un algoritmo genético
    """

    def __init__(self):
        """
        Se incializan variables necesarias para otros métodos de la clase
        """
        print("Implementacion de un algoritmo genetico")
        self.__tiempos__ = []  # para métricas
        self.__tiempos__.append(time.time())  # tiempo inicial
        self.__minimos__ = []  # para métricas
        self.__maximos__ = []  # para métricas
        self.__medias__ = []  # para métricas
        self.__stds__ = []  # para métricas

        self.__fit_pob__ = None  # para el  modelo generacional

    def carga_instancia(self, ruta, sep=';'):
        """
        Carga la instancia del problema. Es independiente al problema
        :param ruta: Ruta con el fichero con la instancia
        :param sep: Separador del fichero plano
        :return:
        """
        instancia = []
        with open(ruta, 'r') as f:
            for linea in f.readlines():
                instancia.append(linea.split(sep))
        self.__n_ejemplos__ = len(instancia)
        self.__n_puntos__ = len(instancia[0])
        self.__instancias__ = instancia

    def elige_instancia(self, indice_instancia=0):
        """
        De las instancias cargadas, selecciona una. Es independiente del problema
        :param indice_instancia: Valor de la instancia a utilizar.
        :return:
        """
        self.__indice_instancia__ = indice_instancia
        self.__instancia__ = self.__instancias__[self.__indice_instancia__]

    def codifica_fenotipo_viajante(self, sep_punt=','):
        """
        Codifica el fenotipo para el problema del problema.
        :param sep_punt: separador de las coordenadas de los puntos en la instancia.
        :return:
        """
        fenotipo = [x.split(sep_punt) for x in self.__instancia__]
        self.__fenotipo__ = {}
        self.__permutaciones__ = []
        i = 0
        for x, y in fenotipo:
            self.__fenotipo__[i] = [float(x), float(y)]
            self.__permutaciones__.append(i)
            i = i + 1

    def genera_poblacion(self, n_pob=1000):
        """
        Genera poblaciones como permutaciones de la representación escogida. Sirve para cualquier problema de adyacencia.
        :param n_pob: número de individuos de la población.
        :return:
        """
        self.__poblacion__ = []
        self.__n_pob__ = n_pob
        l_tmp = self.__permutaciones__.copy()
        for i in range(self.__n_pob__):
            random.shuffle(l_tmp)
            tmp = l_tmp.copy()
            self.__poblacion__.append(tmp)

    def __fit_distancia__(self, permutacion):
        """
        Aplicación de la función de fitness relacionada con la distancia al individuo.
        :param permutacion: Individuo que representa una permutación de la codificación.
        :return:
        """
        posiciones = [self.__fenotipo__[x] for x in permutacion]
        distancias = []
        for i in range(len(posiciones)):
            if i != len(posiciones) - 1:
                distancias.append(np.linalg.norm(np.array(posiciones[i]) - np.array(posiciones[i + 1])))
            else:
                distancias.append(np.linalg.norm(np.array(posiciones[i]) - np.array(posiciones[0])))
        return np.log(1.0 / sum(distancias))

    def __torneo__(self, *individuos):
        """
        Aplica la selección con el método del torneo. Independiente del problema.
        :param individuos: Individuos a los que se les quiere aplicar el torno.
        :return: Devuelve el individuo con mayor fit seguón __fit_distancia__
        """
        distancias = []
        for individuo in individuos:
            distancias.append(self.__fit_distancia__(individuo))
        return individuos[distancias.index(max(distancias))]

    def seleccion_parental(self, k):
        """
        A partir de la población selecciona a los individuos que serán los procreadores.
        :param k: Número de individuos para el torneo
        :return:
        """
        self.__padres__ = []
        for i in range(len(self.__poblacion__)):
            # con reemplazamiento
            self.__padres__.append(self.__torneo__(*random.choices(self.__poblacion__, k=k)))
            # si no queremos reemplazamiento utilizaremos random.sample()

    def __part_map_cross__(self, padre1, padre2):
        """
        Implentación del PMX
        :param padre1: individuo que hará como padre 1
        :param padre2: individuo que hará como padre 2
        :return: dos individuos resultantes de la recombinación
        """
        # Inicializamos los hijos para tener todas las posiciones disponibles.
        hijo1 = [None] * self.__n_puntos__
        hijo2 = [None] * self.__n_puntos__

        # primero seleccionamos dos numeros aleatroios y colocamos el segmento resultante en el hijo.
        punto_1 = int(random.uniform(0, self.__n_puntos__))

        punto_2 = int(random.uniform(0, self.__n_puntos__))

        # Nos aseguramos de coger un segmento y no un único punto igual que el segmento no sea la cadena completa.
        dist = abs(punto_2 - punto_1)

        while (dist <= 2) or ((punto_1 in [0, self.__n_puntos__ - 1]) and (punto_2 in [0, self.__n_puntos__ - 1])):
            punto_2 = int(random.uniform(0, self.__n_puntos__))
            punto_1 = int(random.uniform(0, self.__n_puntos__))
            dist = abs(punto_2 - punto_1)

        # Definimos el segmento que será común para el hijo 1 y el hijo 2
        derecha = max(punto_2, punto_1)
        izquierda = min(punto_2, punto_1)
        index_hijo1 = [i for i in range(izquierda, derecha, 1)]
        index_hijo2 = [i for i in range(izquierda, derecha, 1)]

        # Definimos los índices que quedan fuera del segmento
        index_res1 = [i for i in range(izquierda)]
        index_res1.extend([i for i in range(derecha, self.__n_puntos__)])

        index_res2 = [i for i in range(izquierda)]
        index_res2.extend([i for i in range(derecha, self.__n_puntos__)])

        index_segmento = [i for i in range(izquierda, derecha, 1)]
        val_segmento1 = []
        val_segmento2 = []

        # Hasta aqui index_hijo1 es igual que index_hijo2 que es el segmento. Lo inicializamos con los valores del padre
        # contrario y guardamos el valor del segmento.
        for i in index_segmento:
            hijo1[i] = padre1[i]
            val_segmento1.append(padre1[i])

            hijo2[i] = padre2[i]
            val_segmento2.append(padre2[i])
        # Recorremos los elementos del segmento para ambos descendientes a la vez
        for i in index_segmento:

            # Para el hijo 1
            # Si el valor del padre contrario no está en el segmento generado buscamos relaciones
            if padre2[i] not in val_segmento1:
                i_tmp = i
                # Buscamos la relación entre la posición del padre y el valor del hijo hasta que encontramos un sitio
                # en el que podemos colocar el valor del padre
                while padre2.index(hijo1[i_tmp]) in index_hijo1:
                    i_tmp = padre2.index(hijo1[i_tmp])
                # Actualizamos el valor del hijo
                hijo1[padre2.index(hijo1[i_tmp])] = padre2[i]
                # Actualizamos las posiciones ocupadas por el hijo
                index_hijo1.append(padre2.index(hijo1[i_tmp]))
                try:
                    # En caso de que la posición actualizada estuviera en las restantes del hijo la eliminamos
                    index_res1.pop(index_res1.index(padre2.index(hijo1[i_tmp])))
                except:
                    warnings.warn('Actualizamos segmento interior')

            # Para el hijo 2 acemos lo análogo al hijo 1
            if padre1[i] not in val_segmento2:
                i_tmp = i

                while padre1.index(hijo2[i_tmp]) in index_hijo2:
                    i_tmp = padre1.index(hijo2[i_tmp])
                hijo2[padre1.index(hijo2[i_tmp])] = padre1[i]
                index_hijo2.append(padre1.index(hijo2[i_tmp]))
                try:
                    index_res2.pop(index_res2.index(padre1.index(hijo2[i_tmp])))
                except:
                    warnings.warn('Actualizamos segmento interior')

        # rellenamos el hijo1 con los valores del padre 2 y al revés
        for i in index_res1:
            hijo1[i] = padre2[i]
            index_hijo1.append(i)

        for i in index_res2:
            hijo2[i] = padre1[i]
            index_hijo2.append(i)
        return hijo1, hijo2

    def __swap_mutation__(self, individuo, swap_prob):
        """
        Implementación de la mutación
        :param individuo: Individuo de la población
        :param swap_prob: Probabilidad de que se produzca la mutación
        :return: Devuelve el indivudo mutado
        """
        self.__swap_prob__ = swap_prob
        if swap_prob > 1:
            raise OverflowError('La probabilidad no puede tener valores mayores que 1')
        # Método de la altura, generamos un número aletario y si es menor que la probabilidades hacemos la mutación.
        if random.uniform(0, 1) <= swap_prob:
            punto_1 = int(random.uniform(0, self.__n_puntos__))
            punto_2 = int(random.uniform(0, self.__n_puntos__))

            tmp = individuo[punto_1]
            individuo[punto_1] = individuo[punto_2]
            individuo[punto_2] = tmp

        return individuo

    def __genera_descenencia__(self, swap_prob):
        """
        Aplica la recombinación y la mutación para obtener una descendencia de la recombinación y la mutación
        :param swap_prob: Probablidad de mutación.
        :return:
        """
        self.__descendencia__ = []
        for i in (range(self.__n_pob__ // 2)):
            padre1, padre2 = random.choices(self.__padres__, k=2)
            hijo1, hijo2 = self.__part_map_cross__(padre1, padre2)
            self.__descendencia__.append(self.__swap_mutation__(hijo1, swap_prob=swap_prob))
            self.__descendencia__.append(self.__swap_mutation__(hijo2, swap_prob=swap_prob))
        if self.__n_pob__ % 2 != 0:
            padre1, padre2 = random.choices(self.__padres__, k=2)
            self.__descendencia__.append(
                self.__swap_mutation__(self.__part_map_cross__(padre1, padre2)[0], swap_prob=swap_prob))

    def modelo_generacional(self, swap_prob=0.4, elitismo=True):
        """
        Lleva a cabo toda la creación de una nueva generación. Es independiente del método o problema aplicado ya que
        queda parametrizado en otros módulos.
        :param swap_prob: Probabilidad de mutación.
        :param elitismo: Parámetro que fija si llevar a cabo o no el elitismo
        :return:
        """
        # Generamos la descendencia
        self.__genera_descenencia__(swap_prob=swap_prob)

        # Actualizamos el fitness de la población actual evitando reprocesar
        if type(self.__fit_pob__) is type(None):
            self.__fit_pob__ = [self.__fit_distancia__(x) for x in self.__poblacion__]
        else:
            self.__fit_pob__ = self.__fit_desc__

        self.__fit_desc__ = [self.__fit_distancia__(x) for x in self.__descendencia__]
        # Prácticamos el elitismo si procede
        if elitismo:
            self.__descendencia__[self.__fit_desc__.index(min(self.__fit_desc__))] = \
                self.__poblacion__[self.__fit_pob__.index(max(self.__fit_pob__))]

            self.__fit_desc__[self.__fit_desc__.index(min(self.__fit_desc__))] = \
                self.__fit_pob__[self.__fit_pob__.index(max(self.__fit_pob__))]

        # Actualizamos la población

        self.__poblacion__ = self.__descendencia__.copy()

    def __metricas__(self):
        """
        Cálculo de las métricas de interés
        :return:
        """
        self.__tiempos__.append(time.time())
        self.__minimos__.append(min(self.__fit_desc__))
        self.__maximos__.append(max(self.__fit_desc__))
        self.__medias__.append(np.mean(self.__fit_desc__))
        self.__stds__.append(np.std(self.__fit_desc__))

    def comprueba_descendencia(self):
        """
        Método de debugging que permite conocer si la población está en buene estado.
        :return: 0 si la descendencia cumplee las ocndiciones de integridad.
        """
        media = np.mean(self.__permutaciones__)
        std = np.std(self.__permutaciones__)
        for individuo in self.__descendencia__:
            if media != np.mean(individuo) or std != np.std(individuo):
                raise AttributeError('Error de integridad')
            else:
                return 0

    def escribe_resultados(self, interaccion, repeticion, instancia, data_path=None):
        """
        Escribe los resultados en la dirección especificada
        :param interaccion: Número de interacción
        :param repeticion: Número de repetición con los mismo parámetros
        :param instancia: Instancia
        :param data_path: Ruta a los datos.
        :return:
        """
        self.__metricas__()
        if type(data_path) is type(None):
            self.__resultado_path__ = 'data/CE_resultado_' + str(self.__n_pob__) \
                                      + '_' + str(self.__n_puntos__) \
                                      + '_' + str(self.__swap_prob__) + '.csv'
            if not os.path.exists('data/'):
                os.makedirs('data/')
        else:
            self.__resultado_path__ = data_path
        with open(self.__resultado_path__, 'a') as f:
            f.write(';'.join([str(instancia), str(repeticion), str(interaccion),
                              str(self.__tiempos__[instancia]).replace('.', ','),
                              str(self.__maximos__[instancia]).replace('.', ','),
                              str(self.__minimos__[instancia]).replace('.', ','),
                              str(self.__medias__[instancia]).replace('.', ','),
                              str(self.__stds__[instancia]).replace('.', ',')]) + '\n')

    def solucion(self, criterio='MAX'):
        """
        Devuelve la solución del problema
        :param criterio: flag para saber que individuo devolver.
        :return:
        """
        if criterio == 'MAX':
            return self.__descendencia__[self.__fit_desc__.index(max(self.__fit_desc__))]


class algortimo_evolutivo():
    """
    Esta clase contiene lo necesario para ejecutar un algoritmo evolutivo
    """

    def __init__(self, sd0, modo='gaussian_paso_unico', modo_desc='solo_desc', t=None, modo_muta='un_paso',
                 t_prim=None):
        """
        Se incializan variables necesarias para otros métodos de la clase
        """
        self.__modo__ = modo
        self.__sd__ = sd0
        self.__t__ = t
        self.__t_prim__ = t_prim
        self.__modo_desc__ = modo_desc
        self.__modo_muta__ = modo_muta
        print("Implementacion de un algoritmo evolutivo")
        print([self.__modo__, self.__sd__, self.__t__, self.__t_prim__, self.__modo_desc__, self.__modo_muta__])
        self.__tiempos__ = []  # para métricas
        self.__tiempos__.append(time.time())  # tiempo inicial
        self.__medias__ = []  # para métricas
        self.__cambios__ = 0
        self.__fit__ = None  # para el  modelo generacional

    def define_funcion_optimizar(self, funcion, desplazamiento, lim1, lim2):
        """
        Se define la función a optimizar y se disponibiliza para el resto de etapas
        :param funcion: 'ESFERA' o 'SCHWEFEL'
        :param desplazamiento:
        :param lim1:
        :param lim2:
        :return:
        """
        self.__lim1__ = lim1
        self.__lim2__ = lim2
        self.__desplazamiento__ = desplazamiento
        if funcion == 'ESFERA':
            self.__funcion_opt__ = 'ESFERA'

            def funcion_esfera(variables):
                """
                Ejecuta una esfera desplazada
                :param variables:
                :return:
                """
                return sum([(x - desplazamiento) ** 2 for x in variables])

            self.funcion = funcion_esfera

        if funcion == 'SCHWEFEL':
            def funcion_schwefel(variables):
                return 418.9829 * len(variables) + sum([-1 * x * np.sin(np.sqrt(abs(x))) for x in variables])

            self.funcion = funcion_schwefel

    def genera_poblacion(self, n_pob, n_dim):
        """
        Genera la población para el algoritmo evolutivo
        :param n_pob: Número de individuos en la población
        :param n_dim: Número de dimensiones del problema
        :return:
        """
        self.__n_pob__ = n_pob
        self.__n_dim__ = n_dim
        if self.__modo_muta__ == 'un_paso':
            pob = []
            for i in range(n_pob):
                tmp = []
                for i in range(n_dim):
                    tmp.append(random.random() * np.abs(self.__lim1__ - self.__lim2__) + self.__lim1__)
                pob.append(tmp + [random.random() * self.__sd__])
            self.poblacion = pd.DataFrame(np.stack(pob))
            if type(self.__t__) is type(None):
                self.__t__ = 1. / np.sqrt(n_pob)
            if type(self.__t_prim__) is type(None):
                self.__t_prim__ = 1. / np.sqrt(2 * np.sqrt(n_pob))
        if self.__modo_muta__ == 'n_pasos':
            pob = []
            for i in range(n_pob):
                tmp = []
                tmp2 = []
                for i in range(n_dim):
                    tmp.append(random.random() * np.abs(self.__lim1__ - self.__lim2__) + self.__lim1__)
                    tmp2.append(random.random() * self.__sd__)
                pob.append(tmp + tmp2)
            self.poblacion = pd.DataFrame(np.stack(pob))
            if type(self.__t__) is type(None):
                self.__t__ = 1. / np.sqrt(n_pob)
            if type(self.__t_prim__) is type(None):
                self.__t_prim__ = 1. / np.sqrt(2 * np.sqrt(n_pob))

    def __muta_gaussian__(self, punto, umbral=0.01, mean=0, norm_sd=1):
        """
        Mutación gaussiana
        :param punto: punto a mutar
        :param sd: parámetro sigma
        :param mean: media de la segunda gaussiana
        :param norm_sd: es la desviación estandar de la segunda gaussiana
        :return:
        """
        if self.__modo_muta__ == 'un_paso':
            __sd_new__ = punto[len(punto) - 1] * np.exp(self.__t__ * random.gauss(0, 1))
            if __sd_new__ < umbral:
                __sd_new__ = umbral
            return pd.Series([x + __sd_new__ * random.gauss(mean, norm_sd) for x in punto[:-1]] + [__sd_new__])
        if self.__modo_muta__ == 'n_pasos':
            tmp = [0] * 2 * self.__n_dim__
            for i in range(self.__n_dim__):
                __sd_new__ = punto[i + self.__n_dim__] * np.exp(
                    self.__t__ * random.gauss(0, 1) + self.__t_prim__ * random.gauss(0, 1))
                if __sd_new__ < umbral:
                    __sd_new__ = umbral
                tmp[i + self.__n_dim__] = __sd_new__
                tmp[i] = punto[i] + __sd_new__ * random.gauss(mean, norm_sd)
            return pd.Series(tmp)

    def __recombinacion_discreta__(self, elementos):
        """
        Devuelve los índices para  llevar a cabo una recombinación discreta.
        :param elementos:
        :return:
        """
        rango = list(range(0, elementos))
        n1 = random.randint(0, elementos)
        ind1 = random.sample(rango, n1)
        for i in ind1:
            rango.remove(i)
        return ind1, rango

    def __recombina_series__(self, s1, s2, modo='discreta'):
        """
        Recombina dos series.
        :param s1:
        :param s2:
        :param modo:
        :return:
        """
        if modo == 'discreta':
            if self.__modo_muta__ == 'un_paso':
                id1, id2 = self.__recombinacion_discreta__(self.__n_dim__ + 1)
                return pd.concat([s1.iloc[id1], s2.iloc[id2]], axis=0).sort_index()
            if self.__modo_muta__ == 'n_pasos':
                id1, id2 = self.__recombinacion_discreta__( 2 * self.__n_dim__)
                return pd.concat([s1.iloc[id1], s2.iloc[id2]], axis=0).sort_index()
    def recombina(self, lamb):
        """
        Lleva a cabo la recombinacón de toda la población para generar la descendencia
        :param lamb:
        :return:
        """
        self.__n_desc__ = self.__n_pob__
        self.__n_offs_tot__ = lamb

        df_recombined = pd.DataFrame()
        for i in range(0, self.__n_offs_tot__):
            i1 = random.randint(0, len(self.poblacion) - 1)
            i2 = random.randint(0, len(self.poblacion) - 1)
            tmp = self.__recombina_series__(self.poblacion.iloc[i1, :],
                                            self.poblacion.iloc[i2, :])
            df_recombined = pd.concat([df_recombined, tmp], axis=1)
        df_recombined = df_recombined.T.reset_index(drop=True)
        self.__df_recombined__ = df_recombined

    def muta(self, umbral=0.01, mean=0, norm_sd=1):
        """
        Aplica la mutación
        :param umbral:
        :param mean:
        :param norm_sd:
        :return:
        """
        if self.__modo_muta__ == 'un_paso':
            self.__df_recombined__ = self.__df_recombined__.apply(
                self.__muta_gaussian__, axis=1, args=(umbral, mean, norm_sd))
            self.__df_recombined__.iloc[:, :self.__n_dim__] = self.__df_recombined__.iloc[:, :self.__n_dim__].clip(
                lower=self.__lim1__,
                upper=self.__lim2__)
        if self.__modo_muta__ == 'n_pasos':
            self.__df_recombined__ = self.__df_recombined__.apply(
                self.__muta_gaussian__, axis=1, args=(umbral, mean, norm_sd))
            self.__df_recombined__.iloc[:, :self.__n_dim__] = self.__df_recombined__.iloc[:, :self.__n_dim__].clip(
                lower=self.__lim1__,
                upper=self.__lim2__)

    def genera_descencencia(self, verbose=False):
        """
        Genera la descendencia
        :param verbose:
        :return:
        """
        if self.__modo_desc__ == 'solo_desc':
            if self.__modo_muta__ == 'un_paso':
                self.__df_recombined__.loc[:, self.__n_dim__ + 1] = self.__df_recombined__.iloc[:, :self.__n_dim__].apply(
                    self.funcion,
                    axis=1)
                self.df_desc = self.__df_recombined__.sort_values(by=self.__n_dim__ + 1, ascending=True).iloc[
                               :self.__n_desc__,
                               :self.__n_dim__ + 1].reset_index(drop=True)
                self.__fit_mean_new__ = self.__df_recombined__.sort_values(self.__n_dim__ + 1, ascending=True).iloc[
                                        :self.__n_desc__,
                                        self.__n_dim__ + 1].reset_index(drop=True).mean()
            if self.__modo_muta__ == 'n_pasos':
                self.__df_recombined__.loc[:, 2 * self.__n_dim__] = self.__df_recombined__.iloc[:, :self.__n_dim__].apply(
                    self.funcion,
                    axis=1)
                self.df_desc = self.__df_recombined__.sort_values(by= 2 * self.__n_dim__, ascending=True).iloc[
                               :self.__n_desc__,
                               :2 * self.__n_dim__].reset_index(drop=True)
                self.__fit_mean_new__ = self.__df_recombined__.sort_values(2 * self.__n_dim__, ascending=True).iloc[
                                        :self.__n_desc__,
                                        2 * self.__n_dim__].reset_index(drop=True).mean()

        if self.__modo_desc__ == 'desc_y_padres':
            if self.__modo_muta__ == 'un_paso':
                tmp = pd.concat([self.poblacion, self.__df_recombined__]).reset_index(drop=True)
                tmp.loc[:, self.__n_dim__ + 1] = tmp.iloc[:, :self.__n_dim__].apply(
                    self.funcion,
                    axis=1)
                self.df_desc = tmp.sort_values(by=self.__n_dim__ + 1, ascending=True).iloc[:self.__n_desc__,
                               :self.__n_dim__ + 1].reset_index(drop=True)
                self.__fit_mean_new__ = tmp.sort_values(self.__n_dim__ + 1, ascending=True).iloc[
                                        :self.__n_desc__,
                                        self.__n_dim__ + 1].reset_index(drop=True).mean()
            if self.__modo_muta__ == 'n_pasos':
                tmp = pd.concat([self.poblacion, self.__df_recombined__]).reset_index(drop=True)
                tmp.loc[:,  2 * self.__n_dim__] = tmp.iloc[:, :self.__n_dim__].apply(
                    self.funcion,
                    axis=1)
                self.df_desc = tmp.sort_values(by= 2 * self.__n_dim__, ascending=True).iloc[:self.__n_desc__,
                               : 2 * self.__n_dim__].reset_index(drop=True)
                self.__fit_mean_new__ = tmp.sort_values( 2 * self.__n_dim__, ascending=True).iloc[
                                        :self.__n_desc__,
                                        2 * self.__n_dim__].reset_index(drop=True).mean()
        if verbose:
            print(self.__fit_mean_new__)

    def fin_paso(self, verbose=False):
        """
        En caso de que proceda lleva a cabo el paso de generación
        :param verbose:
        :return:
        """
        if type(self.__fit__) is type(None):
            self.__fit__ = self.__fit_mean_new__
            self.__poblacion__ = self.df_desc
            self.__cambios__ += 1
        elif self.__fit_mean_new__ < self.__fit__:
            if verbose:
                print("NUMERO DE GENERACIONES")
                print(len(self.__tiempos__))
                print("NUMERO DE CAMBIOS")
                print(self.__cambios__ + 1)
                print("FIT NUEVO")
                print(self.__fit_mean_new__)
                print("FIT ANTERIOR")
                print(self.__fit__)
                print(" media de sd ")
                print(self.poblacion.iloc[:, self.__n_dim__].mean())
            self.__fit__ = self.__fit_mean_new__
            self.poblacion = self.df_desc
            self.__cambios__ += 1

        self.__medias__.append(self.__fit__)
        self.__tiempos__.append(time.time())


def ejecuta(ruta, sp, n_pob, n_punt, max_iter):
    """
    Función para la paralelización
    :param sp: Probabilidad de mutación
    :param n_pob: Tamaño de la pobación
    :param n_punt: Número de puntos
    :param max_iter: Límite superior a las iteraciones.
    :return:
    """
    os.system(ruta +
              " ".join([str(sp), str(n_pob), str(n_punt), str(max_iter)]))
