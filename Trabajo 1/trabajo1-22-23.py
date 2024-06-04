# ==========================================================
# Aprendizaje automático
# Máster en Ingeniería Informática - Universidad de Sevilla
# Curso 2021-22
# Primer trabajo práctico
# ===========================================================

# --------------------------------------------------------------------------
# APELLIDOS: Pace
# NOMBRE: Antonio
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo que
# debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite (e incluso se
# recomienda), pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de
# terceros, OBTENIDO A TRAVÉS DE LA RED o cualquier otro medio, se considerará
# plagio.

# Cualquier plagio o compartición de código que se detecte significará
# automáticamente la calificación de CERO EN LA ASIGNATURA para TODOS los
# estudiantes involucrados. Por tanto, NO se les conservará, para
# futuras convocatorias, ninguna nota que hubiesen obtenido hasta el
# momento. SIN PERJUICIO DE OTRAS MEDIDAS DE CARÁCTER DISCIPLINARIO QUE SE
# PUDIERAN TOMAR.
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES Y MÉTODOS
# QUE SE PIDEN


# ========================
# IMPORTANTE: USO DE NUMPY
# ========================

# SE PIDE USAR NUMPY EN LA MEDIDA DE LO POSIBLE.

#
# %%
import numpy as np
from carga_datos import *

# SE PENALIZARÁ el uso de bucles convencionales si la misma tarea se puede
# hacer más eficiente con operaciones entre arrays que proporciona numpy.

# PARTICULARMENTE IMPORTANTE es el uso del método numpy.dot.
# Con numpy.dot podemos hacer productos escalares de pesos por características,
# y extender esta operación de manera compacta a dos dimensiones, cuando tenemos
# varias filas (ejemplos) e incluso varios varios vectores de pesos.

# En lo que sigue, los términos "array" o "vector" se refieren a "arrays de numpy".

# NOTA: En este trabajo NO se permite usar scikit-learn (salvo en el código que
# se proporciona para cargar los datos).

# -----------------------------------------------------------------------------

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aa.zip y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn). Todos los datos se cargan en arrays de numpy.

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresista (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos.

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.

# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn. Como vocabulario,
#   se han usado las 609 palabras que ocurren más frecuentemente en las distintas
#   críticas. Los datos se cargan finalmente en las variables X_train_imdb,
#   X_test_imdb, y_train_imdb,y_test_imdb.

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles).


# ===========================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA (HOLDOUT)
# ===========================================================

# Definir una función

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test, y conservando la correspondencia
# original entre los ejemplos y sus valores de clasificación.
# La división ha de ser ALEATORIA y ESTRATIFICADA respecto del valor de clasificación.

# %%
def particion_entr_prueba(X, y, test=0.20, random_state=None):

    train_indices = []
    test_indices = []

    # dividing the data in train and test sets, preserving the original class proportions
    for i in np.unique(y):
        class_indices = np.where(y == i)[0]
        np.random.shuffle(class_indices)

        num_test_samples = int(test * len(class_indices))
        test_indices.extend(class_indices[:num_test_samples])
        train_indices.extend(class_indices[num_test_samples:])

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:


# %%
Xe_votos, Xp_votos, ye_votos, yp_votos = particion_entr_prueba(
    X_votos, y_votos, test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:

y_votos.shape[0], ye_votos.shape[0], yp_votos.shape[0]
# Out: (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# In[3]: np.unique(y_votos,return_counts=True)
# Out[3]: (array(['democrata', 'republicano'], dtype='<U11'), array([267, 168]))
# In[4]: np.unique(ye_votos,return_counts=True)
# Out[4]: (array(['democrata', 'republicano'], dtype='<U11'), array([178, 112]))
# In[5]: np.unique(yp_votos,return_counts=True)
# Out[5]: (array(['democrata', 'republicano'], dtype='<U11'), array([89, 56]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con más de dos clases:

# %%
Xe_credito, Xp_credito, ye_credito, yp_credito = particion_entr_prueba(
    X_credito, y_credito, test=0.4)

np.unique(y_credito, return_counts=True)
# Out[7]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([202, 228, 220]))

# In[8]: np.unique(ye_credito,return_counts=True)
# Out[8]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([121, 137, 132]))

# In[9]: np.unique(yp_credito,return_counts=True)
# Out[9]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([81, 91, 88]))
# ------------------------------------------------------------------

# just a simple function for partition data into train and test sets permutating data.


# ===========================================
# EJERCICIO 2: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# Se pide implementar el clasificador de regresión logística mini-batch
# a través de una clase python, que ha de tener la siguiente estructura:

# class RegresionLogisticaMiniBatch():

#    def __init__(self,normalizacion=False,
#                 rate=0.1,rate_decay=False,batch_tam=64,
#                 pesos_iniciales=None):

#          .....

#    def entrena(self,entr,clas_entr,n_epochs=1000,
#                reiniciar_pesos=False):

#         ......

#     def clasifica_prob(self,E):


#         ......

#     def clasifica(self,E):


#         ......


# Explicamos a continuación cada uno de los métodos:


# * Constructor de la clase:
# --------------------------

#  El constructor debe tener los siguientes argumentos de entrada:


#  - El parámetro normalizacion, que puede ser True o False (False por
#    defecto). Indica si los datos se tienen que normalizar, tanto para el
#    entrenamiento como para la clasificación de nuevas instancias.  La
#    normalización es una técnica que suele ser útil cuando los distintos
#    atributos reflejan cantidades numéricas de muy distinta magnitud.
#    En ese caso, antes de entrenar se calcula la media m_i y la desviación
#    típica d_i en CADA COLUMNA i (es decir, en cada atributo) de los
#    datos del conjunto de entrenamiento.  A continuación, y antes del
#    entrenamiento, esos datos se transforman de manera que cada componente
#    x_i se cambia por (x_i - m_i)/d_i. Esta MISMA transformación se realiza
#    sobre las nuevas instancias que se quieran clasificar.

#  - rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  - rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula:
#       rate_n= (rate_0)*(1/(1+n))
#    donde n es el número de epoch, y rate_0 es la cantidad
#    introducida en el parámetro rate anterior.

#  - batch_tam: indica el tamaño de los mini batches (por defecto 64)
#    que se usan para calcular cada actualización de pesos.

#  - pesos_iniciales: Si es None, los pesos iniciales se inician
#    aleatoriamente. Si no, debe proporcionar un array de pesos que se
#    tomarán como pesos iniciales.

#

# * Método entrena:
# -----------------

#  Este método es el que realiza el entrenamiento del clasificador.
#  Debe calcular un vector de pesos, mediante el correspondiente
#  algoritmo de entrenamiento basado en ascenso por el gradiente mini-batch,
#  para maximizar la log verosimilitud. Describimos a continuación los parámetros de
#  entrada:

#  - entr y clas_entr, son los datos del conjunto de entrenamiento y su
#    clasificación, respectivamente. El primero es un array (bidimensional)
#    con los ejemplos, y el segundo un array (unidimensional) con las clasificaciones
#    de esos ejemplos, en el mismo orden.

#  - n_epochs: número de pasadas que se realizan sobre todo el conjunto de
#    entrenamiento.

#  - reiniciar_pesos: si es True, se reinicia al comienzo del
#    entrenamiento el vector de pesos de manera aleatoria
#    (típicamente, valores aleatorios entre -1 y 1).
#    Si es False, solo se inician los pesos la primera vez que se
#    llama a entrena. En posteriores veces, se parte del vector de
#    pesos calculado en el entrenamiento anterior. Esto puede ser útil
#    para continuar el aprendizaje a partir de un aprendizaje
#    anterior, si por ejemplo se dispone de nuevos datos.

#  NOTA: El entrenamiento en mini-batch supone que en cada epoch se
#  recorren todos los ejemplos del conjunto de entrenamiento,
#  agrupados en grupos del tamaño indicado. Por cada uno de estos
#  grupos de ejemplos se produce una actualización de los pesos.
#  Se pide una VERSIÓN ESTOCÁSTICA, en la que en cada epoch se asegura que
#  se recorren todos los ejemplos del conjunto de entrenamiento,
#  en un orden ALEATORIO, aunque agrupados en grupos del tamaño indicado.


# * Método clasifica_prob:
# ------------------------

#  Método que devuelve el array de correspondientes probabilidades de pertenecer
#  a la clase positiva (la que se ha tomado como clase 1), para cada ejemplo de un
#  array E de nuevos ejemplos.


# * Método clasifica:
# -------------------

#  Método que devuelve un array con las correspondientes clases que se predicen
#  para cada ejemplo de un array E de nuevos ejemplos. La clase debe ser una de las
#  clases originales del problema (por ejemplo, "republicano" o "democrata" en el
#  problema de los votos).


# Si el clasificador aún no ha sido entrenado, tanto "clasifica" como
# "clasifica_prob" deben devolver una excepción del siguiente tipo:

# %%
class ClasificadorNoEntrenado(Exception):
    pass


class RegresionLogisticaMiniBatch():
    def __init__(self, normalizacion=False, rate=0.1, rate_decay=False, batch_tam=64, pesos_iniciales=None):
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.pesos_iniciales = pesos_iniciales
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_weights(self, n):
        return np.random.rand(n, 1)

    def normalizar(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std

    def entrena(self, entr, clas_entr, n_epochs=1000, reiniciar_pesos=False):
        if self.normalizacion:
            entr = self.normalizar(entr)

        if reiniciar_pesos or self.weights is None:
            self.weights = self.pesos_iniciales if self.pesos_iniciales is not None else self.initialize_weights(
                entr.shape[1])

        m = entr.shape[0]

        # convert categorical classes to binary values
        unique_classes = np.unique(clas_entr)
        self.classes_mapping = {label: idx for idx,
                                label in enumerate(unique_classes)}
        clas_entr = np.vectorize(self.classes_mapping.get)(
            clas_entr).reshape(-1, 1)

        for epoch in range(1, n_epochs + 1):
            if self.rate_decay:
                current_rate = self.rate / (1 + epoch)
            else:
                current_rate = self.rate

            # random order of examples
            indexes = np.random.permutation(m)
            X_batch = entr[indexes]
            y_batch = clas_entr[indexes]

            # iterate over mini batches covering all examples
            for i in range(0, m, self.batch_tam):
                end_idx = i + self.batch_tam if i + self.batch_tam < m else m
                X_mini_batch = X_batch[i:end_idx]
                y_mini_batch = y_batch[i:end_idx]

                predictions = self.sigmoid(X_mini_batch.dot(self.weights))
                error = y_mini_batch - predictions
                gradient = X_mini_batch.T.dot(error) / self.batch_tam

                self.weights += current_rate * gradient

    def clasifica_prob(self, E):
        if self.weights is None:
            raise ClasificadorNoEntrenado(
                "Classifier not trained. Call 'entrena' before 'clasifica_prob'.")
        if self.normalizacion:
            E = self.normalizar(E)
        return self.sigmoid(E.dot(self.weights))

    def clasifica(self, E):
        prob = self.clasifica_prob(E)

        # reverse mapping from binary values to categorical classes
        categorie_ordinate = list(self.classes_mapping.keys())

        # return the class rounding the probability
        pred = [categorie_ordinate[valore_binario]
                for valore_binario in (prob >= 0.5).astype(int).flatten()]

        return pred

# Ejemplos de uso:
# ----------------


# CON LOS DATOS VOTOS:

#

# En primer lugar, separamos los datos en entrenamiento y prueba (los resultados pueden
# cambiar, ya que esta partición es aleatoria)

Xe_votos, Xp_votos, ye_votos, yp_votos = particion_entr_prueba(
    X_votos, y_votos)

# Creamos el clasificador:

RLMB_votos = RegresionLogisticaMiniBatch()

# Lo entrenamos sobre los datos de entrenamiento:

RLMB_votos.entrena(Xe_votos, ye_votos)

# Con el clasificador aprendido, realizamos la predicción de las clases
# de los datos que estan en test:
RLMB_votos.clasifica_prob(Xp_votos)
# array([3.90234132e-04, 1.48717603e-11, 3.90234132e-04, 9.99994374e-01, 9.99347533e-01,...])

RLMB_votos.clasifica(Xp_votos)
# Out[5]: array(['democrata', 'democrata', 'democrata','republicano',... ], dtype='<U11')

# Calculamos la proporción de aciertos en la predicción, usando la siguiente
# función que llamaremos "rendimiento".


def rendimiento(clasif, X, y):
    return sum(clasif.clasifica(X) == y)/y.shape[0]


print(rendimiento(RLMB_votos, Xp_votos, yp_votos))
# Out[6]: 0.9080459770114943

# ---------------------------------------------------------------------

# CON LOS DATOS DEL CÀNCER

# Hacemos un experimento similar al anterior, pero ahora con los datos del
# cáncer de mama, y usando normalización y disminución de la tasa


Xe_cancer, Xp_cancer, ye_cancer, yp_cancer = particion_entr_prueba(
    X_cancer, y_cancer)


RLMB_cancer = RegresionLogisticaMiniBatch(normalizacion=True, rate_decay=True)

RLMB_cancer.entrena(Xe_cancer, ye_cancer)

RLMB_cancer.clasifica_prob(Xp_cancer)
# Out[9]: array([9.85046885e-01, 8.77579844e-01, 7.81826115e-07,..])

RLMB_cancer.clasifica(Xp_cancer)
# Out[10]: array([1, 1, 0,...])

print(rendimiento(RLMB_cancer, Xp_cancer, yp_cancer))
# Out[11]: 0.9557522123893806


# =================================================
# EJERCICIO 3: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================

# Este ejercicio vale 2 PUNTOS (SOBRE 10) pero se puede saltar, sin afectar
# al resto del trabajo. Puede servir para el ajuste de parámetros en los ejercicios
# posteriores, pero si no se realiza, se podrían ajustar siguiendo el método "holdout"
# implementado en el ejercicio 1.

# La técnica de validación cruzada que se pide en este ejercicio se explica
# en el tema "Evaluación de modelos".

# Definir una función:

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador. El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cáncer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad,
# no tiene por qué coincidir exactamente el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#             {"batch_tam":16,"rate_decay":True},Xe_cancer,ye_cancer,n=5)
# 0.9121095227289917

# %%
def rendimiento_validacion_cruzada(clase_clasificador, params, X, y, n=5, epochs=1000):
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    fold_size = m // n

    accuracies = []

    # for each fold we train the classifier with the rest of the folds and validate it with the current fold, storing accuracy
    for i in range(n):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n - 1 else m

        val_indices = indices[start:end]
        X_val = X[val_indices]
        y_val = y[val_indices]

        train_indices = np.concatenate([indices[:start], indices[end:]])
        X_train = X[train_indices]
        y_train = y[train_indices]

        clasificador = clase_clasificador(**params)
        clasificador.entrena(X_train, y_train, n_epochs=epochs)

        accuracy = rendimiento(clasificador, X_val, y_val)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)

    return mean_accuracy

# Ejemplo de uso:
# rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                     {"batch_tam": 16, "rate_decay": True},
#                                     Xe_cancer, ye_cancer, n=5)


# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones deben ser aleatorias y estratificadas.

# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:
# LR16 = RegresionLogisticaMiniBatch(batch_tam=16, rate_decay=True)
# LR16.entrena(Xe_cancer, ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# rendimiento(LR16, Xp_cancer, yp_cancer)
# 0.9203539823008849

# ------------------------------------------------------------------------------


# ===================================================
# EJERCICIO 4: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================


# Usando el modelo implementado en el ejercicio 2, obtener clasificadores
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama
# - Críticas de películas en IMDB

# Ajustar los parámetros para mejorar el rendimiento. Si se ha hecho el ejercicio 3,
# usar validación cruzada para el ajuste (si no, usar el "holdout" del ejercicio 1).

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos.

# %%
CV_cancer = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                           {"batch_tam": 32, "rate": 0.01,
                                               "rate_decay": False},
                                           Xe_cancer, ye_cancer, n=5)

CV_cancer2 = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                            {"batch_tam": 32, "rate": 0.0001,
                                             "rate_decay": False},
                                            Xe_cancer, ye_cancer, n=5)

CV_cancer3 = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                            {"batch_tam": 16, "rate": 0.1,
                                             "rate_decay": True},
                                            Xe_cancer, ye_cancer, n=5)

print("CV cancer: ", CV_cancer)
print("CV cancer2: ", CV_cancer2)
print("CV cancer3: ", CV_cancer3)
# out: CV cancer:  0.9035117056856187
# out: CV cancer2:  0.8837075967510749
# out: CV cancer3:  0.9209746774964167

# %%
CV_votos = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                          {"batch_tam": 32, "rate": 0.1,
                                              "rate_decay": True},
                                          Xe_votos, ye_votos, n=5)

CV_votos2 = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                           {"batch_tam": 32, "rate": 0.001,
                                            "rate_decay": False},
                                           Xe_votos, ye_votos, n=8)


CV_votos3 = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                           {"batch_tam": 32, "rate": 0.01,
                                            "rate_decay": False},
                                           Xe_votos, ye_votos, n=5)

print("CV votos: ", CV_votos)
print("CV votos2: ", CV_votos2)
print("CV votos3: ", CV_votos3)
# out: CV votos:  0.890942028985507
# out: CV votos2:  0.9244186046511628
# out: CV votos3:  0.9541062801932367

# %%
CV_imdb = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                         {"batch_tam": 256, "rate": 0.1,
                                          "rate_decay": True},
                                         X_train_imdb, y_train_imdb, n=5)

CV_imdb2 = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                          {"batch_tam": 128, "rate": 0.01,
                                           "rate_decay": False},
                                          X_train_imdb, y_train_imdb, n=5)
print("CV imdb: ", CV_imdb)
print("CV imdb2: ", CV_imdb2)

# =====================================
# EJERCICIO 5: CLASIFICACIÓN MULTICLASE
# =====================================

# Técnica "One vs Rest" (Uno frente al Resto)
# -------------------------------------------


# Se pide implementar la técnica "One vs Rest" (Uno frente al Resto),
# para obtener un clasificador multiclase a partir del clasificador
# binario definido en el apartado anterior.


#  En concreto, se pide implementar una clase python
#  RegresionLogisticaOvR con la siguiente estructura, y que implemente
#  el entrenamiento y la clasificación siguiendo el método "One vs
#  Rest" tal y como se ha explicado en las diapositivas del módulo.


# class RegresionLogisticaOvR():

#    def __init__(self,normalizacion=False,rate=0.1,rate_decay=False,
#                 batch_tam=64):

#          .....

#    def entrena(self,entr,clas_entr,n_epochs=1000):

#         ......

#    def clasifica(self,E):


#         ......


#  Los parámetros de los métodos significan lo mismo que en el
#  apartado anterior.

# %%
class RegresionLogisticaOvR():
    def __init__(self, normalizacion=False, rate=0.1, rate_decay=False, batch_tam=64):
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.classifiers = {}
        self.classes_mapping = None

    def entrena(self, entr, clas_entr, n_epochs=1000, reiniciar_pesos=False):

        unique_classes = np.unique(clas_entr)

        self.classes_mapping = {label: idx for idx,
                                label in enumerate(unique_classes)}
        # for each distinct class we train a classifier
        for target_class in unique_classes:
            binary_clas_entr = (clas_entr == target_class).astype(int)
            classifier = RegresionLogisticaMiniBatch(
                normalizacion=self.normalizacion,
                rate=self.rate,
                rate_decay=self.rate_decay,
                batch_tam=self.batch_tam
            )
            classifier.entrena(entr, binary_clas_entr,
                               n_epochs, reiniciar_pesos)
            self.classifiers[target_class] = classifier

    def clasifica_prob(self, E):
        if not self.classifiers:
            raise ClasificadorNoEntrenado(
                "Classifier not trained. Call 'entrena' before 'clasifica_prob'.")

        # matrix of probabilities
        # for each row of E, we have a probability for each classifier
        prob_matrix = np.zeros((E.shape[0], len(self.classifiers)))

        for idx, (target_class, classifier) in enumerate(self.classifiers.items()):
            prob_matrix[:, idx] = classifier.clasifica_prob(E).flatten()

        return prob_matrix

    def clasifica(self, E):
        prob_matrix = self.clasifica_prob(E)

        # max element index (class with highest probability) for each row
        class_indices = np.argmax(prob_matrix, axis=1)

        categorie_ordinate = list(self.classes_mapping.keys())
        pred = [categorie_ordinate[idx] for idx in class_indices]
        return pred

#  Un ejemplo de sesión, con el problema del iris:

# --------------------------------------------------------------------


Xe_iris, Xp_iris, ye_iris, yp_iris = particion_entr_prueba(
    X_iris, y_iris, test=1/3)

rl_iris = RegresionLogisticaOvR(rate=0.01, batch_tam=20)

rl_iris.entrena(Xe_iris, ye_iris)
print("ovr: ", rendimiento(rl_iris, Xe_iris, ye_iris))
# 0.97

print("ovr: ", rendimiento(rl_iris, Xp_iris, yp_iris))
# >>> 0.94
# --------------------------------------------------------------------


# ==============================================
# EJERCICIO 6: APLICACION A PROBLEMAS MULTICLASE
# ==============================================


# ---------------------------------------------------------
# 6.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación del apartado anterior, para obtener un
# clasificador que aconseje la concesión, estudio o no concesión de un préstamo,
# basado en los datos X_credito, y_credito. Ajustar adecuadamente los parámetros.

# NOTA IMPORTANTE: En este caso concreto, los datos han de ser transformados,
# ya que los atributos de este conjunto de datos no son numéricos. Para ello, usar la llamada
# "codificación one-hot", descrita en el tema "Preprocesado e ingeniería de características".
# Se pide implementar esta transformación (directamete, SIN USAR Scikt Learn ni Pandas).

# %%
def one_hot_encode(X):
    categorical_columns = [col for col in range(
        X.shape[1]) if not np.issubdtype(X[:, col].dtype, np.number)]

    category_mappings = {}

    for col in categorical_columns:
        unique_categories = np.unique(X[:, col])
        category_mapping = {category: idx for idx,
                            category in enumerate(unique_categories)}
        category_mappings[col] = category_mapping
        X[:, col] = np.vectorize(category_mapping.get)(X[:, col])

    one_hot_encoded = np.column_stack([np.eye(len(np.unique(X[:, col])))[
                                      X[:, col].astype(int)] for col in categorical_columns])

    X_encoded = np.delete(X, categorical_columns, axis=1)
    X_encoded = np.concatenate(
        (X_encoded.astype(np.float64), one_hot_encoded), axis=1)

    return X_encoded


Xe_credito, Xp_credito, ye_credito, yp_credito = particion_entr_prueba(
    X_credito, y_credito, test=1/3)

rl_credito = RegresionLogisticaOvR(rate=0.01, batch_tam=20)

Xe_credito_encoded = one_hot_encode(Xe_credito)
Xp_credito_encoded = one_hot_encode(Xp_credito)

rl_credito.entrena(Xe_credito_encoded, ye_credito)

print("OvR credito: ", rendimiento(rl_credito, Xe_credito_encoded, ye_credito))
# 0.78

# ---------------------------------------------------------
# 6.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación o implementaciones del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador.

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test).


# load data, lines is a list of strings, transformed in a 2d numpy array
# %%
def load_digit_data(file_path):
    lines = open(file_path, 'r').read().splitlines()
    images = np.array([np.array(list(c)) for c in lines])
    images = (images != ' ').astype(int)
    return images

# load labels, just a list of int


def load_labels_data(file_path):
    lines = open(file_path, 'r').read().splitlines()
    labels = np.array(lines).astype(int)
    return labels


# load digits from file
Xe_digits = load_digit_data('datos/digitdata/trainingimages')
ye_digits = load_labels_data('datos/digitdata/traininglabels')

Xp_digits = load_digit_data('datos/digitdata/testimages')
yp_digits = load_labels_data('datos/digitdata/testlabels')

Xv_digits = load_digit_data('datos/digitdata/validationimages')
yv_digits = load_labels_data('datos/digitdata/validationlabels')

# reshape in an array of 28x28 flattened images
Xe_digits = Xe_digits.reshape(-1, 28 * 28)
Xp_digits = Xp_digits.reshape(-1, 28 * 28)
Xv_digits = Xv_digits.reshape(-1, 28 * 28)

rl_digits = RegresionLogisticaOvR(rate=0.1, batch_tam=256)

rl_digits.entrena(Xe_digits, ye_digits)

print("OvR digits: ", rendimiento(
    rl_digits, Xp_digits, yp_digits))
# 0.96

# %%
