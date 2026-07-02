Listo. Te dejo el **resumen completo de LIME y SHAP con las fórmulas en formato simple**, para que puedas copiar y pegar en Word sin que se rompan. Está basado en los PDFs de LIME y SHAP.  

---

# 1. LIME

LIME significa **Local Interpretable Model-Agnostic Explanations**. Es un método de explicabilidad que permite explicar **una predicción individual** de un modelo complejo. Es **local**, porque explica el comportamiento del modelo cerca de un dato específico, y es **modelo-agnóstico**, porque puede aplicarse a cualquier modelo mientras se puedan consultar sus predicciones.

La idea principal es usar un modelo complejo `f`, que queremos explicar, y aproximarlo localmente con un modelo simple `g`.

```text
f = modelo complejo que queremos explicar
g = modelo simple usado como explicación
x = dato original que queremos explicar
x' = representación interpretable del dato
```

La idea central es:

```text
g ≈ f cerca del dato x
```

Es decir, LIME no intenta explicar todo el modelo, sino solo cómo se comporta alrededor de una predicción específica.

---

## Representación interpretable

LIME convierte el dato original en una representación más fácil de entender.

En texto:

```text
x' = vector de palabras presentes o ausentes
```

En imágenes:

```text
x' = vector de superpíxeles presentes o ausentes
```

Por ejemplo, si una imagen tiene 4 superpíxeles:

```text
x' = [1, 1, 1, 1]
```

Eso significa que todos los superpíxeles están presentes.

Una perturbación podría ser:

```text
z' = [1, 0, 1, 0]
```

Eso significa que se mantienen los superpíxeles 1 y 3, mientras que los superpíxeles 2 y 4 se ocultan o atenúan.

---

## Pipeline de LIME

El proceso completo de LIME se puede resumir así:

```text
x → x' → z' → z → f(z) → g → explicación
```

En palabras:

1. Se toma el dato original `x`.
2. Se crea una representación interpretable `x'`.
3. Se generan perturbaciones `z'`.
4. Se reconstruyen como datos reales `z`.
5. Se evalúa el modelo complejo `f(z)`.
6. Se entrena un modelo simple `g`.
7. Los pesos de `g` se usan como explicación.

---

## Modelo explicador lineal

LIME suele usar un modelo lineal simple como explicación:

```text
g(z') = b + w1*z1' + w2*z2' + ... + wk*zk'
```

Donde:

```text
g(z') = predicción del modelo explicador
b = intercepto
wj = peso de la característica j
zj' = 1 si la característica j está presente
zj' = 0 si la característica j está ausente
```

En una imagen, cada `wj` puede representar la importancia de un superpíxel.

---

## Peso de cercanía

LIME no trata todas las perturbaciones igual. Las perturbaciones más parecidas al dato original pesan más. Para eso usa un kernel de proximidad:

```text
pi_x(z) = exp( - D(x,z)^2 / sigma^2 )
```

Donde:

```text
pi_x(z) = peso de cercanía de la perturbación z
D(x,z) = distancia entre el dato original x y la perturbación z
sigma = parámetro que controla qué tan rápido baja el peso
```

La idea es:

```text
Si z se parece mucho a x, pi_x(z) es alto.
Si z es muy distinto de x, pi_x(z) es bajo.
```

Esto hace que la explicación sea local.

---

## Pérdida local ponderada

LIME entrena el modelo simple `g` para que imite al modelo complejo `f`, pero dando más importancia a las perturbaciones cercanas al dato original.

```text
L(f,g,pi_x) = sum pi_x(z) * ( f(z) - g(z') )^2
```

Donde:

```text
f(z) = predicción del modelo complejo para la perturbación z
g(z') = predicción del modelo explicador simple
pi_x(z) = peso de cercanía
```

En simple:

```text
LIME quiere que g(z') se parezca a f(z), especialmente para perturbaciones cercanas a x.
```

---

## Objetivo general de LIME

LIME busca una explicación que sea fiel al modelo complejo, pero también simple:

```text
xi(x) = argmin_g [ L(f,g,pi_x) + Omega(g) ]
```

Donde:

```text
xi(x) = explicación final para el dato x
L(f,g,pi_x) = error local entre f y g
Omega(g) = penalización por complejidad del modelo explicador
```

En palabras:

```text
LIME busca un modelo simple que copie bien al modelo complejo cerca de x, pero que no sea demasiado complejo.
```

---

## Sparse Linear Explanations

LIME busca explicaciones dispersas, es decir, explicaciones que usen pocas características importantes.

Por ejemplo:

```text
Si una imagen tiene 100 superpíxeles, no queremos explicar usando los 100.
Queremos usar solo los K superpíxeles más relevantes.
```

Esto hace que la explicación sea más entendible.

---

## Lasso y K-Lasso en LIME

Lasso ayuda a seleccionar variables importantes, porque puede hacer que algunos coeficientes queden en cero.

En LIME, K-Lasso se usa para seleccionar solo `K` características relevantes.

```text
Omega(g) = infinito si numero_de_pesos_distintos_de_cero > K
Omega(g) = 0 si numero_de_pesos_distintos_de_cero <= K
```

Donde:

```text
K = número máximo de características que se mostrarán en la explicación
```

La idea es:

```text
La explicación debe usar como máximo K variables, palabras o superpíxeles.
```

---

## SP-LIME

SP-LIME significa **Submodular Pick LIME**.

LIME explica una instancia individual. SP-LIME sirve cuando queremos elegir varias explicaciones representativas para entender mejor el modelo completo.

La pregunta que responde SP-LIME es:

```text
Si solo puedo revisar B explicaciones, ¿cuáles debería mirar?
```

Para eso construye una matriz de explicaciones `W`.

```text
W_ij = peso de la característica j en la explicación de la instancia i
```

La importancia global de una característica se puede escribir como:

```text
I_j = sum_i W_ij
```

Donde:

```text
I_j = importancia global de la característica j
W_ij = peso de la característica j en la instancia i
```

Luego SP-LIME busca cubrir la mayor cantidad posible de características importantes:

```text
c(V,W,I) = sum_j indicador[existe i en V tal que W_ij > 0] * I_j
```

Donde:

```text
V = conjunto de instancias seleccionadas
W = matriz de explicaciones
I_j = importancia global de la característica j
```

El problema de selección es:

```text
Pick(W,I) = argmax_V c(V,W,I), sujeto a que |V| <= B
```

Donde:

```text
B = número máximo de explicaciones que una persona está dispuesta a revisar
```

En simple:

```text
SP-LIME selecciona pocas explicaciones, pero intentando que sean diversas, representativas y no redundantes.
```

---

## Ejemplo breve de LIME

Supongamos que una imagen tiene 3 superpíxeles:

```text
x' = [1, 1, 1]
```

El modelo complejo predice:

```text
f(x) = 0.90
```

LIME genera perturbaciones:

```text
[1, 1, 1] → f(z) = 0.90
[0, 1, 1] → f(z) = 0.55
[1, 0, 1] → f(z) = 0.82
[1, 1, 0] → f(z) = 0.65
```

Interpretación:

```text
Al quitar S1, la predicción baja de 0.90 a 0.55.
Al quitar S2, la predicción baja de 0.90 a 0.82.
Al quitar S3, la predicción baja de 0.90 a 0.65.
```

Entonces `S1` parece ser el más importante, luego `S3`, y finalmente `S2`.

LIME podría aprender un modelo simple como:

```text
g(z') = 0.10 + 0.35*z1' + 0.08*z2' + 0.27*z3'
```

La explicación sería:

```text
S1 > S3 > S2
```

Es decir:

```text
La predicción se explica principalmente por los superpíxeles S1 y S3.
```

---

# 2. SHAP

SHAP significa **SHapley Additive Explanations**. Es un método de explicabilidad basado en los **Shapley Values** de la teoría de juegos.

Su objetivo es repartir la predicción del modelo entre las variables de forma justa.

---

## Idea central de SHAP

SHAP explica una predicción como una suma:

```text
f(x) = phi_0 + phi_1 + phi_2 + ... + phi_M
```

O también:

```text
f(x) = phi_0 + sum_i phi_i
```

Donde:

```text
phi_0 = valor base
phi_i = valor SHAP de la variable i
M = número total de variables
```

La idea es:

```text
La predicción final se obtiene partiendo desde un valor base y sumando o restando los aportes de cada variable.
```

---

## Modelo explicativo aditivo

SHAP pertenece a los métodos de atribución aditiva de características. Su modelo explicativo tiene esta forma:

```text
g(z') = phi_0 + phi_1*z1' + phi_2*z2' + ... + phi_M*zM'
```

Donde:

```text
z_i' = 1 si la variable i está presente
z_i' = 0 si la variable i está ausente
phi_i = aporte de la variable i
```

---

## Interpretación de los valores SHAP

Cada variable tiene un valor SHAP:

```text
phi_i
```

Si:

```text
phi_i > 0
```

la variable aumenta la predicción.

Si:

```text
phi_i < 0
```

la variable disminuye la predicción.

Si:

```text
phi_i ≈ 0
```

la variable casi no influye en esa predicción.

---

## Shapley Values: idea desde teoría de juegos

SHAP trata cada variable como si fuera un jugador.

```text
Jugador → variable del modelo
Coalición → subconjunto de variables
Ganancia → predicción del modelo
Aporte del jugador → aporte de una variable
Shapley Value → valor SHAP
```

La contribución marginal de una variable se calcula así:

```text
contribucion_marginal_i = v(S union {i}) - v(S)
```

Donde:

```text
S = conjunto de variables ya presentes
i = variable que se agrega
v(S) = predicción usando las variables del conjunto S
```

En palabras:

```text
La contribución marginal mide cuánto cambia la predicción cuando agrego la variable i.
```

---

## Fórmula de Shapley Value

La fórmula general es:

```text
phi_i = sum_{S subseteq F\{i}} [ |S|! * (|F|-|S|-1)! / |F|! ] * [ v(S union {i}) - v(S) ]
```

Versión más legible:

```text
phi_i = suma de las contribuciones marginales de i en todos los subconjuntos posibles, ponderadas según su frecuencia.
```

Donde:

```text
F = conjunto de todas las variables
S = subconjunto de variables que no contiene a i
|S| = número de variables en S
|F| = número total de variables
v(S union {i}) - v(S) = contribución marginal de i
```

---

## Propiedades de SHAP

SHAP se apoya en tres propiedades principales.

### 1. Local accuracy

Los valores SHAP deben sumar la predicción final:

```text
f(x) = phi_0 + sum_i phi_i
```

Esto significa:

```text
La explicación reconstruye exactamente la predicción del modelo para ese dato.
```

---

### 2. Missingness

Si una variable está ausente, su contribución debe ser cero.

```text
si z_i' = 0, entonces phi_i*z_i' = 0
```

En simple:

```text
Una variable ausente no debe aportar a la explicación.
```

---

### 3. Consistency

Si en un nuevo modelo una variable aporta más a la predicción, su valor SHAP no debería disminuir.

Ejemplo:

```text
Modelo f: edad cambia la predicción en +10%
Modelo f': edad cambia la predicción en +20%
```

Entonces esperamos:

```text
SHAP_edad en f' >= SHAP_edad en f
```

En simple:

```text
Si una variable se vuelve más importante para el modelo, su atribución no debería bajar.
```

---

## Problema de calcular SHAP exacto

Calcular SHAP exacto puede ser muy costoso, porque se deben considerar muchos subconjuntos de variables.

Si hay `M` variables, hay:

```text
2^M subconjuntos posibles
```

Por ejemplo:

```text
M = 20 → 2^20 = 1.048.576 subconjuntos
```

Por eso, en la práctica se usan aproximaciones.

---

## Variables ausentes y valor esperado condicional

En Machine Learning no es trivial “apagar” una variable, porque el modelo necesita una entrada completa.

Por eso SHAP usa la idea de valor esperado condicional:

```text
f_x(z') = f(h_x(z')) = E[ f(z) | z_S ]
```

Donde:

```text
z_S = variables conocidas o presentes
E[ f(z) | z_S ] = valor esperado de la predicción cuando conocemos solo las variables de S
```

En palabras:

```text
Si algunas variables están ausentes, SHAP estima la predicción promedio considerando posibles valores para esas variables faltantes.
```

Ejemplo:

```text
E[ f(z) | edad = 65, IMC = 30 ]
```

Significa:

```text
Predicción promedio del modelo para casos donde conocemos edad e IMC, promediando sobre las variables ausentes.
```

---

## Kernel SHAP

Kernel SHAP es una variante modelo-agnóstica que combina ideas de Linear LIME con Shapley Values.

Su kernel es:

```text
pi_x(z') = (M - 1) / [ C(M, |z'|) * |z'| * (M - |z'|) ]
```

Donde:

```text
M = número total de variables
|z'| = número de variables presentes en la coalición
C(M, |z'|) = combinatoria: formas de elegir |z'| variables entre M
```

Kernel SHAP funciona así:

1. Muestrea coaliciones:

```text
z'_k ∈ {0,1}^M
```

2. Reconstruye instancias en el espacio original usando un background dataset.

3. Evalúa el modelo complejo:

```text
f(z_k)
```

4. Ajusta una regresión lineal ponderada.

5. Los coeficientes de esa regresión son los valores SHAP:

```text
phi_i
```

En simple:

```text
Kernel SHAP usa una regresión ponderada especial para obtener valores SHAP aproximados.
```

---

## Variantes de SHAP

El PDF menciona varias variantes:

```text
KernelExplainer = modelo-agnóstico, útil para distintos modelos, pero puede ser costoso.
TreeExplainer = eficiente para modelos basados en árboles, como Random Forest o XGBoost.
DeepExplainer = usado para redes neuronales profundas, basado en DeepLIFT + Shapley Values.
```

---

## Ejemplo breve de SHAP

Supongamos un modelo que predice riesgo de enfermedad.

El valor base es:

```text
phi_0 = 0.40
```

Para un paciente específico, SHAP entrega:

```text
phi_edad = +0.20
phi_IMC = +0.12
phi_presion = -0.07
```

Entonces:

```text
f(x) = phi_0 + phi_edad + phi_IMC + phi_presion
```

Reemplazando:

```text
f(x) = 0.40 + 0.20 + 0.12 - 0.07
```

Resultado:

```text
f(x) = 0.65
```

Interpretación:

```text
El modelo partía desde un riesgo base de 0.40.
La edad aumentó la predicción en 0.20.
El IMC aumentó la predicción en 0.12.
La presión normal redujo la predicción en 0.07.
Por eso la predicción final fue 0.65.
```

---

# 3. Comparación LIME vs SHAP

```text
LIME:
- Explica una predicción individual.
- Es local.
- Es modelo-agnóstico.
- Genera perturbaciones alrededor del dato original.
- Entrena un modelo simple g para aproximar localmente al modelo complejo f.
- Usa pesos de cercanía pi_x(z).
- Puede usar K-Lasso para seleccionar pocas variables.
- SP-LIME permite elegir explicaciones representativas para entender mejor el modelo.
```

```text
SHAP:
- Explica una predicción individual.
- Es local.
- Se basa en Shapley Values.
- Reparte la predicción entre las variables de forma justa.
- Usa valores phi_i para indicar cuánto aporta cada variable.
- Cumple propiedades como local accuracy, missingness y consistency.
- Calcular SHAP exacto puede ser caro.
- Kernel SHAP aproxima los valores SHAP usando regresión ponderada.
```

---

# Idea final para recordar

```text
LIME = aproxima localmente un modelo complejo usando un modelo simple.
```

```text
SHAP = reparte la predicción entre las variables usando contribuciones justas.
```

Más completo:

```text
LIME perturba el dato, observa cómo cambia la predicción y entrena un explicador local.
```

```text
SHAP calcula cuánto aporta cada variable a la predicción, considerando todas las coaliciones posibles.
```


Claro. Este PDF trata sobre **métodos contrafactuales en XAI**, es decir, explicaciones que no solo dicen “qué variable fue importante”, sino que responden algo más accionable: **qué debería cambiar en el dato de entrada para que el modelo entregue otra predicción**. Las slides se basan principalmente en Wachter et al. 2017, Dandl et al. 2020, Mothilal et al. 2020 y Guidotti 2022. 

La idea central de los métodos contrafactuales es preguntar:

```text
¿Qué podría hacerse de manera diferente para lograr un resultado diferente?
```

Por ejemplo, si un modelo rechaza un crédito bancario, una explicación contrafactual no diría solamente “el ingreso fue importante” o “las deudas influyeron”, sino algo como:

```text
Si el ingreso mensual aumentara en X, 
si la deuda bajara en Y,
y si el número de tarjetas fuera Z,
entonces el crédito habría sido aprobado.
```

Eso es distinto de LIME o SHAP. LIME y SHAP explican **por qué el modelo tomó una decisión** en términos de importancia o contribución de variables. En cambio, los contrafactuales explican **qué cambios mínimos harían que la decisión cambiara**.

En imágenes, la idea es similar. Si el modelo clasifica una imagen como “maligna”, un contrafactual busca una versión modificada de esa imagen que sea lo más parecida posible a la original, pero que el modelo clasifique como “benigna”. La explicación no es una lista de pesos, sino una comparación entre la imagen original y una imagen modificada que cambia la predicción.

---

Un contrafactual se puede pensar así:

```text
x = dato original
f(x) = predicción original del modelo
x' = dato contrafactual
f(x') = predicción deseada
```

La meta es encontrar un `x'` que cumpla dos cosas al mismo tiempo:

```text
1. f(x') debe ser la salida deseada.
2. x' debe parecerse lo más posible a x.
```

Es decir, queremos cambiar la predicción, pero modificando lo mínimo posible el dato original. Por eso los contrafactuales son explicaciones locales: explican una instancia específica, no todo el modelo.

Un buen contrafactual debería cumplir cuatro condiciones importantes:

```text
1. Lograr la predicción deseada.
2. Ser lo más parecido posible al dato original.
3. Cambiar pocas variables.
4. Tener valores realistas.
```

Por ejemplo, si a Patricia le rechazaron un crédito, no sería útil decirle:

```text
Si tu edad fuera -5 años, el crédito sería aprobado.
```

Eso cambia la predicción, pero no es realista. Tampoco sería útil decir:

```text
Si cambiaras 25 variables al mismo tiempo, el crédito sería aprobado.
```

Eso puede ser correcto matemáticamente, pero poco accionable. Una buena explicación contrafactual debe ser cercana, realista y útil.

---

El primer método importante del PDF es el de **Wachter et al. 2017**. Este método propone buscar una instancia contrafactual resolviendo un problema de optimización. La fórmula general, escrita de forma simple, es:

```text
min_x' max_lambda [ lambda * ( f_w(x') - y' )^2 + d(x_i, x') ]
```

Donde:

```text
f_w = modelo ML con parámetros fijos
x_i = instancia original
x' = instancia contrafactual que buscamos
y' = salida deseada
d(x_i, x') = distancia entre el dato original y el contrafactual
lambda = parámetro que controla la importancia de alcanzar la salida deseada
```

La fórmula tiene dos partes principales.

La primera parte es:

```text
lambda * ( f_w(x') - y' )^2
```

Esto castiga que la predicción del contrafactual esté lejos de la salida deseada. Si queremos que el modelo prediga “aprobado”, pero todavía predice “rechazado”, este término será grande.

La segunda parte es:

```text
d(x_i, x')
```

Esto castiga que el contrafactual sea muy distinto del dato original. Queremos que `x'` cambie lo menos posible respecto de `x_i`.

Entonces el método intenta equilibrar dos objetivos:

```text
Cambiar la predicción
pero modificando lo menos posible el dato original.
```

El parámetro `lambda` ayuda a evitar que el método encuentre una solución muy parecida al dato original, pero que no cambie realmente la predicción. Si la predicción todavía no se acerca a `y'`, aumentar `lambda` hace que ese error pese más, obligando al algoritmo a buscar un contrafactual que sí logre el resultado deseado.

---

Wachter también propone una distancia específica: la **distancia de Manhattan ponderada por la inversa del MAD**. En formato simple:

```text
d(x_i, x') = sum_k | x_i,k - x'_k | / MAD_k
```

Donde:

```text
x_i,k = valor de la variable k en el dato original
x'_k = valor de la variable k en el contrafactual
MAD_k = desviación mediana absoluta de la variable k
```

El MAD se define como:

```text
MAD_k = mediana_j | x_j,k - mediana_l(x_l,k) |
```

La idea del MAD es normalizar las variables según su variabilidad. Si una variable normalmente varía mucho, un cambio grande en esa variable no debería penalizarse tanto. Si una variable normalmente varía poco, un cambio pequeño puede ser más relevante.

Por ejemplo, cambiar el ingreso mensual en 50.000 pesos puede ser un cambio pequeño si los ingresos varían mucho en el dataset. Pero cambiar el número de hijos de 1 a 5 puede ser un cambio mucho más fuerte. La distancia ponderada ayuda a que las variables se comparen de forma más justa.

El método de Wachter cumple bien dos cosas:

```text
1. Busca una predicción deseada.
2. Busca que el contrafactual sea cercano al dato original.
```

Pero tiene limitaciones:

```text
1. No garantiza múltiples explicaciones diversas.
2. No garantiza que el contrafactual sea realista.
```

Por ejemplo, podría encontrar solo una solución, o podría modificar valores de manera poco plausible.

---

Luego aparece el método de **Dandl et al. 2020**, que mejora la idea anterior usando una optimización multiobjetivo. En vez de tener solo “lograr la salida deseada” y “parecerse al dato original”, Dandl agrega más criterios.

La idea es encontrar un contrafactual `x` para una instancia original `x*` que cumpla:

```text
1. Que f(x) esté cerca de la salida deseada.
2. Que x sea cercano a x*.
3. Que x difiera de x* en pocas variables.
4. Que x sea realista respecto de los datos observados.
```

La función multiobjetivo se puede escribir de forma simple como:

```text
min_x o(x) = min_x ( o1(f(x), Y'), o2(x, x*), o3(x, x*), o4(x, X_obs) )
```

Cada término mide algo distinto.

El primer término mide si el contrafactual logra la salida deseada:

```text
o1(f(x), Y')
```

Si la predicción ya pertenece al conjunto de salidas deseadas `Y'`, el costo es 0. Si no, se penaliza qué tan lejos está de lograrlo.

El segundo término mide la cercanía entre el contrafactual y el dato original usando la **distancia de Gower**:

```text
o2(x, x*) = (1/p) * sum_j delta_G(x_j, x*_j)
```

Donde:

```text
p = número de variables
delta_G = distancia de Gower por variable
```

Para variables numéricas:

```text
delta_G(x_j, x*_j) = |x_j - x*_j| / R_j
```

Donde `R_j` es el rango observado de la variable `j`.

Para variables categóricas:

```text
delta_G(x_j, x*_j) = 1 si x_j es distinto de x*_j
delta_G(x_j, x*_j) = 0 si x_j es igual a x*_j
```

La distancia de Gower es útil porque permite comparar variables numéricas y categóricas dentro de una misma medida.

El tercer término mide cuántas variables cambiaron. Esto se relaciona con la norma `l0`, que cuenta cuántos elementos son distintos:

```text
o3(x, x*) = número de variables que cambiaron
```

Esto es importante porque una explicación contrafactual debería ser simple. Es mejor decir:

```text
Cambia estas 2 variables.
```

que decir:

```text
Cambia estas 15 variables.
```

El cuarto término mide qué tan realista es el contrafactual:

```text
o4(x, X_obs)
```

La idea es comparar el contrafactual con puntos reales observados en el dataset. Si el contrafactual queda muy lejos de los datos reales, probablemente no es plausible. Dandl usa una distancia promedio ponderada de Gower con los `k` puntos observados más cercanos. En simple:

```text
Si x está cerca de datos reales observados, es más realista.
Si x está lejos de todos los datos reales, es menos realista.
```

Como esta optimización tiene varios objetivos al mismo tiempo, se usa un algoritmo evolutivo llamado **NSGA-II**, que permite encontrar soluciones que equilibran estos criterios sin reducir todo a un único número.

---

Después aparece **DICE**, que significa **Diverse Counterfactual Explanations**. Este método se centra especialmente en generar **múltiples contrafactuales diversos**.

La idea es que no siempre existe una única forma de cambiar una predicción. Por ejemplo, si a una persona le rechazan un crédito, podría haber varias alternativas:

```text
Opción 1: aumentar ingreso.
Opción 2: reducir deuda.
Opción 3: aumentar ingreso y reducir número de tarjetas.
```

DICE intenta generar un conjunto de contrafactuales:

```text
c1, c2, ..., ck
```

donde cada `c_i` cambia la predicción del modelo, pero además las soluciones son diversas entre sí.

La función de pérdida de DICE, en formato simple, es:

```text
C(x) = argmin_{c1,...,ck} 
       (1/k) * sum_i yloss(f(c_i), y)
       + (lambda1/k) * sum_i dist(c_i, x)
       - lambda2 * dpp_diversity(c1,...,ck)
```

Tiene tres partes principales.

La primera parte:

```text
(1/k) * sum_i yloss(f(c_i), y)
```

castiga que los contrafactuales no logren la clase deseada.

La segunda parte:

```text
(lambda1/k) * sum_i dist(c_i, x)
```

castiga que los contrafactuales sean muy distintos del dato original.

La tercera parte:

```text
- lambda2 * dpp_diversity(c1,...,ck)
```

premia la diversidad. Tiene signo negativo porque estamos minimizando, entonces restar diversidad equivale a favorecer soluciones diversas.

El `yloss` puede ser una hinge loss. En formato simple:

```text
hinge_loss = max(0, 1 - z * logit(f(c)))
```

Donde:

```text
z = 1 si la clase deseada es 1
z = -1 si la clase deseada es 0
f(c) = probabilidad predicha por el modelo
logit(f(c)) = log( f(c) / (1 - f(c)) )
```

La intuición es esta:

```text
Si el contrafactual ya logra la clase deseada, la pérdida es 0.
Si no logra la clase deseada, la pérdida es positiva.
```

Por ejemplo, si quiero que el contrafactual sea clase 1 y el modelo predice 0.9, eso está bien y no se penaliza. Pero si quiero que sea clase 0 y el modelo predice 0.9, eso está mal y se penaliza mucho.

Para medir diversidad, DICE usa una idea basada en **Determinantal Point Processes**, abreviado DPP. La métrica de diversidad se basa en el determinante de una matriz `K`:

```text
dpp_diversity = det(K)
```

Donde:

```text
K_ij = 1 / (1 + dist(c_i, c_j))
```

Si dos contrafactuales son muy parecidos, su distancia es pequeña, entonces `K_ij` queda cerca de 1. Eso hace que las filas de la matriz se parezcan mucho y el determinante tienda a ser pequeño. En cambio, si los contrafactuales son distintos, la matriz tiene filas más independientes y el determinante puede ser mayor.

En simple:

```text
DICE busca contrafactuales que sean válidos, cercanos al dato original y distintos entre sí.
```

---

La última parte del PDF trata sobre **contrafactuales generativos**, especialmente usando GANs. Esto aparece porque los métodos clásicos como Wachter o DICE pueden funcionar bien en datos tabulares, pero en imágenes pueden tener problemas. Si se modifican directamente los píxeles, pueden aparecer imágenes poco realistas, con artefactos visuales o cambios que no respetan la estructura real de los datos.

Por ejemplo, en imágenes médicas, modificar píxel por píxel puede generar una imagen que cambia la predicción del modelo, pero que no parece una imagen médica real. Eso limita mucho la utilidad de la explicación.

Para resolver esto, se propone generar contrafactuales en el **espacio latente** de un modelo generativo.

Primero hay que entender qué es una **GAN**, o Generative Adversarial Network. Una GAN tiene dos componentes:

```text
Generador = crea imágenes falsas a partir de ruido.
Discriminador = intenta distinguir imágenes reales de imágenes generadas.
```

El generador aprende a producir imágenes cada vez más realistas para engañar al discriminador. El discriminador aprende a detectar si una imagen es real o falsa. Ambos se entrenan en competencia.

El flujo básico es:

```text
z ~ N(0,1) → Generador → imagen generada → Discriminador → probabilidad de ser real
```

Donde `z` es un vector latente, es decir, una representación interna de menor dimensión que codifica características importantes de la imagen.

La idea de los contrafactuales generativos es no modificar directamente los píxeles, sino modificar el vector latente. En vez de hacer esto:

```text
imagen original → modificar píxeles → imagen contrafactual
```

se hace esto:

```text
imagen original → codificación latente → modificar vector latente → generar imagen contrafactual
```

Esto tiene varias ventajas:

```text
1. Las imágenes son más realistas.
2. Los cambios son más suaves.
3. Se respeta mejor la distribución de los datos.
4. La explicación puede ser más interpretable visualmente.
```

---

El PDF menciona como ejemplo **CheXplaining**, un método que genera contrafactuales en imágenes médicas usando StyleGAN. La idea es buscar cambios en el espacio latente que modifiquen la predicción del clasificador, pero manteniendo imágenes realistas.

El flujo general de CheXplaining es:

```text
1. Se recibe una imagen de entrada I.
2. Un encoder transforma la imagen en un vector latente w.
3. Un clasificador preentrenado predice la clase de la imagen.
4. StyleGAN usa el vector latente para generar imágenes.
5. Se buscan direcciones en el espacio latente que cambien la salida del clasificador.
6. Se generan imágenes contrafactuales realistas.
```

Un punto importante es que CheXplaining no hace optimización directa como Wachter ni búsqueda aleatoria. En vez de eso, intenta aprender direcciones interpretables en el espacio latente, llamadas direcciones en el **StyleSpace**, que están alineadas con el comportamiento del clasificador.

En simple:

```text
CheXplaining busca qué dirección en el espacio latente transforma una imagen para cambiar la predicción del modelo, manteniendo la imagen realista.
```

Esto es especialmente relevante para imágenes, porque en imágenes no basta con cambiar valores numéricos: el resultado debe seguir pareciendo una imagen plausible.

---

La idea final del tema es que los métodos contrafactuales explican modelos de una forma muy distinta a LIME y SHAP. Mientras LIME y SHAP responden:

```text
¿Qué partes o variables influyeron en esta predicción?
```

los contrafactuales responden:

```text
¿Qué tendría que cambiar para obtener otra predicción?
```

Por eso son muy intuitivos y accionables. En un crédito, pueden decir qué cambiar para ser aprobado. En imágenes médicas, pueden mostrar qué transformación haría que el modelo cambiara de clase. Pero también tienen desafíos importantes: los cambios deben ser mínimos, realistas, diversos y factibles.

La frase clave para recordar es:

```text
Un contrafactual es una versión modificada del dato original que cambia la predicción del modelo, intentando cambiar lo menos posible y mantenerse realista.
```
