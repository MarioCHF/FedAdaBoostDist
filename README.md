# FedAdaBoostDist. Un algoritmo federado.

En este repositorio se encuentra el código de FedAdaBoostDist, que es una implementación de ADaBoost federado usando destilación de conocimiento. En concreto consiste en lo siguiente:

En cada iteración, cada cliente entrena un clasificador base con una muestra de sus datos en función de sus pesos asociados, en la primera iteración uniforme. Cada cliente luego usa sus modelos para predecir un conjunto de datos sin etiquetar, que es el mismo para todos ellos. Estas predicciones son
usadas en un proceso de destilación de conocimiento por el servidor, que entrena un clasificador que envía a los clientes. Este clasificador es utilizado para actualizar los pesos de los datos de los clientes que serán usados en la siguiente iteración,
de forma que se centre más en aquellos ejemplos mal clasificados hasta el momento. El repositorio está organizado de la siguiente forma:

En la carpeta models se encuentran los ficheros correspondientes a los modelos usados para los experimentos, tanto para la nueva propuesta, FedAdaBoostDist, como para los algoritmos usados para los modelos locales (AdaBoost.M1) y para los modelos del estado del 
arte (FRF, DistBoost.F, PreWeak.F y AdaBoost.F). Respecto a los modelos del estado del arte, FRF es adaptado de la librería flextrees y el resto son adaptados del código del artículo original (https://ieeexplore.ieee.org/document/9892284).

En la carpeta experiments se encuentran cuadernos con dos experimentos realizados, con sus correspondientes tests estadísticos. El resto de experimentos pueden ser reproducidos de forma similar. 

En la carpeta utils se encuentran herramientas para la evaluación y visualización de resultados.

Por último, se ha incluido un cuaderno ejemplo para mostrar cómo ejecutar tanto FedAdaBoostDist como las adaptaciones del estado del arte y obtener sus métricas.
