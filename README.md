# Proyecto-Final-IA1-
Este proyecto es una implementa un algoritmo de optimización conocido como Differential Evolution (DE) para encontrar la posición óptima de una plantilla dentro de una imagen más grande, utilizando la medida de correlación normalizada (NCC) como función de aptitud. Aquí hay algunas conclusiones sobre el código y su aplicación:



Este código implementa un algoritmo de optimización llamado "Differential Evolution" (DE) para encontrar la posición de un objeto (representado por una plantilla o template) dentro de una imagen más grande. El algoritmo utiliza la medida de correlación normalizada (NCC) como función de aptitud (fitness) para evaluar qué tan bien se superpone la plantilla con la región actualmente propuesta en la imagen.

Aquí está la explicación detallada del código:

Función de Correlación Normalizada (NCC):
La función NCC calcula la correlación normalizada entre la plantilla y una región de la imagen dada una posición (x, y) en la imagen. Se utiliza para evaluar qué tan bien coincide la plantilla con la región de la imagen en esa posición.

Algoritmo Differential Evolution (DE):
El algoritmo DE es un algoritmo de optimización evolutiva que busca encontrar la posición óptima para la plantilla en la imagen. Aquí están los pasos clave:

Inicialización:

n_Gen: Número de generaciones.
n_Pop: Tamaño de la población.
dim: Dimensión del espacio de búsqueda (en este caso, 2 para las coordenadas x e y).
Inicialización de la Población:

Se genera una población de soluciones aleatorias dentro de los límites definidos por lb (límites inferiores) y ub (límites superiores).
Evaluación Inicial:

Se calcula el valor de aptitud (NCC) para cada solución inicial.
Optimización:

El bucle principal itera a través de las generaciones.
Para cada individuo en la población, se realiza la selección de vectores de muestra (r1, r2, r3) y se genera un vector mutante v.
Se realiza la recombinación con la solución actual para producir un nuevo vector u.
Se aplica una penalización para asegurar que las soluciones estén dentro de los límites definidos.
Se compara la aptitud de la nueva solución con la solución actual, y si es mejor, se actualiza la posición.
Se registra la mejor aptitud en cada generación.
Visualización (Opcional):

Si animacion es verdadero, se visualiza la evolución de las soluciones en cada generación en una animación.
Resultados:

Se grafica la evolución de la mejor aptitud a lo largo de las generaciones.
Se devuelve la mejor solución encontrada.
Visualización del Resultado Final:
Después de la optimización, se muestra la imagen original con un rectángulo rojo que indica la posición y tamaño de la plantilla encontrada.

Nota:
La eficacia del algoritmo depende en gran medida de la elección adecuada de los parámetros como n_Gen, n_Pop, F, y Cr. Ajustar estos parámetros puede ser necesario según el problema específico que estés abordando.




