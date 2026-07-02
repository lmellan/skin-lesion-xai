Listo. Te dejo el resumen completo de LIME y SHAP con las fórmulas en formato simple, para que puedas copiar y pegar en Word sin que se rompan. Está basado en los PDFs de LIME y SHAP.

1. LIME
LIME significa Local Interpretable Model-Agnostic Explanations. Es un método de explicabilidad que permite explicar una predicción individual de un modelo complejo. Es local, porque explica el comportamiento del modelo cerca de un dato específico, y es modelo-agnóstico, porque puede aplicarse a cualquier modelo mientras se puedan consultar sus predicciones.

La idea principal es usar un modelo complejo f, que queremos explicar, y aproximarlo localmente con un modelo simple g.

f = modelo complejo que queremos explicar
g = modelo simple usado como explicación
x = dato original que queremos explicar
x' = representación interpretable del dato
La idea central es:

g ≈ f cerca del dato x
Es decir, LIME no intenta explicar todo el modelo, sino solo cómo se comporta alrededor de una predicción específica.

Representación interpretable
LIME convierte el dato original en una representación más fácil de entender.

En texto:

x' = vector de palabras presentes o ausentes
En imágenes:

x' = vector de superpíxeles presentes o ausentes
Por ejemplo, si una imagen tiene 4 superpíxeles:

x' = [1, 1, 1, 1]
Eso significa que todos los superpíxeles están presentes.

Una perturbación podría ser:

z' = [1, 0, 1, 0]
Eso significa que se mantienen los superpíxeles 1 y 3, mientras que los superpíxeles 2 y 4 se ocultan o atenúan.

Pipeline de LIME
El proceso completo de LIME se puede resumir así:

x → x' → z' → z → f(z) → g → explicación
En palabras:

Se toma el dato original x.
Se crea una representación interpretable x'.
Se generan perturbaciones z'.
Se reconstruyen como datos reales z.
Se evalúa el modelo complejo f(z).
Se entrena un modelo simple g.
Los pesos de g se usan como explicación.
Modelo explicador lineal
LIME suele usar un modelo lineal simple como explicación:

g(z') = b + w1*z1' + w2*z2' + ... + wk*zk'
Donde:

g(z') = predicción del modelo explicador
b = intercepto
wj = peso de la característica j
zj' = 1 si la característica j está presente
zj' = 0 si la característica j está ausente
En una imagen, cada wj puede representar la importancia de un superpíxel.

Peso de cercanía
LIME no trata todas las perturbaciones igual. Las perturbaciones más parecidas al dato original pesan más. Para eso usa un kernel de proximidad:

pi_x(z) = exp( - D(x,z)^2 / sigma^2 )
Donde:

pi_x(z) = peso de cercanía de la perturbación z
D(x,z) = distancia entre el dato original x y la perturbación z
sigma = parámetro que controla qué tan rápido baja el peso
La idea es:

Si z se parece mucho a x, pi_x(z) es alto.
Si z es muy distinto de x, pi_x(z) es bajo.
Esto hace que la explicación sea local.

Pérdida local ponderada
LIME entrena el modelo simple g para que imite al modelo complejo f, pero dando más importancia a las perturbaciones cercanas al dato original.

L(f,g,pi_x) = sum pi_x(z) * ( f(z) - g(z') )^2
Donde:

f(z) = predicción del modelo complejo para la perturbación z
g(z') = predicción del modelo explicador simple
pi_x(z) = peso de cercanía
En simple:

LIME quiere que g(z') se parezca a f(z), especialmente para perturbaciones cercanas a x.
Objetivo general de LIME
LIME busca una explicación que sea fiel al modelo complejo, pero también simple:

xi(x) = argmin_g [ L(f,g,pi_x) + Omega(g) ]
Donde:

xi(x) = explicación final para el dato x
L(f,g,pi_x) = error local entre f y g
Omega(g) = penalización por complejidad del modelo explicador
En palabras:

LIME busca un modelo simple que copie bien al modelo complejo cerca de x, pero que no sea demasiado complejo.
Sparse Linear Explanations
LIME busca explicaciones dispersas, es decir, explicaciones que usen pocas características importantes.

Por ejemplo:

Si una imagen tiene 100 superpíxeles, no queremos explicar usando los 100.
Queremos usar solo los K superpíxeles más relevantes.
Esto hace que la explicación sea más entendible.

Lasso y K-Lasso en LIME
Lasso ayuda a seleccionar variables importantes, porque puede hacer que algunos coeficientes queden en cero.

En LIME, K-Lasso se usa para seleccionar solo K características relevantes.

Omega(g) = infinito si numero_de_pesos_distintos_de_cero > K
Omega(g) = 0 si numero_de_pesos_distintos_de_cero <= K
Donde:

K = número máximo de características que se mostrarán en la explicación
La idea es:

La explicación debe usar como máximo K variables, palabras o superpíxeles.
SP-LIME
SP-LIME significa Submodular Pick LIME.

LIME explica una instancia individual. SP-LIME sirve cuando queremos elegir varias explicaciones representativas para entender mejor el modelo completo.

La pregunta que responde SP-LIME es:

Si solo puedo revisar B explicaciones, ¿cuáles debería mirar?
Para eso construye una matriz de explicaciones W.

W_ij = peso de la característica j en la explicación de la instancia i
La importancia global de una característica se puede escribir como:

I_j = sum_i W_ij
Donde:

I_j = importancia global de la característica j
W_ij = peso de la característica j en la instancia i
Luego SP-LIME busca cubrir la mayor cantidad posible de características importantes:

c(V,W,I) = sum_j indicador[existe i en V tal que W_ij > 0] * I_j
Donde:

V = conjunto de instancias seleccionadas
W = matriz de explicaciones
I_j = importancia global de la característica j
El problema de selección es:

Pick(W,I) = argmax_V c(V,W,I), sujeto a que |V| <= B
Donde:

B = número máximo de explicaciones que una persona está dispuesta a revisar
En simple:

SP-LIME selecciona pocas explicaciones, pero intentando que sean diversas, representativas y no redundantes.
Ejemplo breve de LIME
Supongamos que una imagen tiene 3 superpíxeles:

x' = [1, 1, 1]
El modelo complejo predice:

f(x) = 0.90
LIME genera perturbaciones:

[1, 1, 1] → f(z) = 0.90
[0, 1, 1] → f(z) = 0.55
[1, 0, 1] → f(z) = 0.82
[1, 1, 0] → f(z) = 0.65
Interpretación:

Al quitar S1, la predicción baja de 0.90 a 0.55.
Al quitar S2, la predicción baja de 0.90 a 0.82.
Al quitar S3, la predicción baja de 0.90 a 0.65.
Entonces S1 parece ser el más importante, luego S3, y finalmente S2.

LIME podría aprender un modelo simple como:

g(z') = 0.10 + 0.35*z1' + 0.08*z2' + 0.27*z3'
La explicación sería:

S1 > S3 > S2
Es decir:

La predicción se explica principalmente por los superpíxeles S1 y S3.
2. SHAP
SHAP significa SHapley Additive Explanations. Es un método de explicabilidad basado en los Shapley Values de la teoría de juegos.

Su objetivo es repartir la predicción del modelo entre las variables de forma justa.

Idea central de SHAP
SHAP explica una predicción como una suma:

f(x) = phi_0 + phi_1 + phi_2 + ... + phi_M
O también:

f(x) = phi_0 + sum_i phi_i
Donde:

phi_0 = valor base
phi_i = valor SHAP de la variable i
M = número total de variables
La idea es:

La predicción final se obtiene partiendo desde un valor base y sumando o restando los aportes de cada variable.
Modelo explicativo aditivo
SHAP pertenece a los métodos de atribución aditiva de características. Su modelo explicativo tiene esta forma:

g(z') = phi_0 + phi_1*z1' + phi_2*z2' + ... + phi_M*zM'
Donde:

z_i' = 1 si la variable i está presente
z_i' = 0 si la variable i está ausente
phi_i = aporte de la variable i
Interpretación de los valores SHAP
Cada variable tiene un valor SHAP:

phi_i
Si:

phi_i > 0
la variable aumenta la predicción.

Si:

phi_i < 0
la variable disminuye la predicción.

Si:

phi_i ≈ 0
la variable casi no influye en esa predicción.

Shapley Values: idea desde teoría de juegos
SHAP trata cada variable como si fuera un jugador.

Jugador → variable del modelo
Coalición → subconjunto de variables
Ganancia → predicción del modelo
Aporte del jugador → aporte de una variable
Shapley Value → valor SHAP
La contribución marginal de una variable se calcula así:

contribucion_marginal_i = v(S union {i}) - v(S)
Donde:

S = conjunto de variables ya presentes
i = variable que se agrega
v(S) = predicción usando las variables del conjunto S
En palabras:

La contribución marginal mide cuánto cambia la predicción cuando agrego la variable i.
Fórmula de Shapley Value
La fórmula general es:

phi_i = sum_{S subseteq F\{i}} [ |S|! * (|F|-|S|-1)! / |F|! ] * [ v(S union {i}) - v(S) ]
Versión más legible:

phi_i = suma de las contribuciones marginales de i en todos los subconjuntos posibles, ponderadas según su frecuencia.
Donde:

F = conjunto de todas las variables
S = subconjunto de variables que no contiene a i
|S| = número de variables en S
|F| = número total de variables
v(S union {i}) - v(S) = contribución marginal de i
Propiedades de SHAP
SHAP se apoya en tres propiedades principales.

1. Local accuracy
Los valores SHAP deben sumar la predicción final:

f(x) = phi_0 + sum_i phi_i
Esto significa:

La explicación reconstruye exactamente la predicción del modelo para ese dato.
2. Missingness
Si una variable está ausente, su contribución debe ser cero.

si z_i' = 0, entonces phi_i*z_i' = 0
En simple:

Una variable ausente no debe aportar a la explicación.
3. Consistency
Si en un nuevo modelo una variable aporta más a la predicción, su valor SHAP no debería disminuir.

Ejemplo:

Modelo f: edad cambia la predicción en +10%
Modelo f': edad cambia la predicción en +20%
Entonces esperamos:

SHAP_edad en f' >= SHAP_edad en f
En simple:

Si una variable se vuelve más importante para el modelo, su atribución no debería bajar.
Problema de calcular SHAP exacto
Calcular SHAP exacto puede ser muy costoso, porque se deben considerar muchos subconjuntos de variables.

Si hay M variables, hay:

2^M subconjuntos posibles
Por ejemplo:

M = 20 → 2^20 = 1.048.576 subconjuntos
Por eso, en la práctica se usan aproximaciones.

Variables ausentes y valor esperado condicional
En Machine Learning no es trivial “apagar” una variable, porque el modelo necesita una entrada completa.

Por eso SHAP usa la idea de valor esperado condicional:

f_x(z') = f(h_x(z')) = E[ f(z) | z_S ]
Donde:

z_S = variables conocidas o presentes
E[ f(z) | z_S ] = valor esperado de la predicción cuando conocemos solo las variables de S
En palabras:

Si algunas variables están ausentes, SHAP estima la predicción promedio considerando posibles valores para esas variables faltantes.
Ejemplo:

E[ f(z) | edad = 65, IMC = 30 ]
Significa:

Predicción promedio del modelo para casos donde conocemos edad e IMC, promediando sobre las variables ausentes.
Kernel SHAP
Kernel SHAP es una variante modelo-agnóstica que combina ideas de Linear LIME con Shapley Values.

Su kernel es:

pi_x(z') = (M - 1) / [ C(M, |z'|) * |z'| * (M - |z'|) ]
Donde:

M = número total de variables
|z'| = número de variables presentes en la coalición
C(M, |z'|) = combinatoria: formas de elegir |z'| variables entre M
Kernel SHAP funciona así:

Muestrea coaliciones:
z'_k ∈ {0,1}^M
Reconstruye instancias en el espacio original usando un background dataset.

Evalúa el modelo complejo:

f(z_k)
Ajusta una regresión lineal ponderada.

Los coeficientes de esa regresión son los valores SHAP:

phi_i
En simple:

Kernel SHAP usa una regresión ponderada especial para obtener valores SHAP aproximados.
Variantes de SHAP
El PDF menciona varias variantes:

KernelExplainer = modelo-agnóstico, útil para distintos modelos, pero puede ser costoso.
TreeExplainer = eficiente para modelos basados en árboles, como Random Forest o XGBoost.
DeepExplainer = usado para redes neuronales profundas, basado en DeepLIFT + Shapley Values.
Ejemplo breve de SHAP
Supongamos un modelo que predice riesgo de enfermedad.

El valor base es:

phi_0 = 0.40
Para un paciente específico, SHAP entrega:

phi_edad = +0.20
phi_IMC = +0.12
phi_presion = -0.07
Entonces:

f(x) = phi_0 + phi_edad + phi_IMC + phi_presion
Reemplazando:

f(x) = 0.40 + 0.20 + 0.12 - 0.07
Resultado:

f(x) = 0.65
Interpretación:

El modelo partía desde un riesgo base de 0.40.
La edad aumentó la predicción en 0.20.
El IMC aumentó la predicción en 0.12.
La presión normal redujo la predicción en 0.07.
Por eso la predicción final fue 0.65.
3. Comparación LIME vs SHAP
LIME:
- Explica una predicción individual.
- Es local.
- Es modelo-agnóstico.
- Genera perturbaciones alrededor del dato original.
- Entrena un modelo simple g para aproximar localmente al modelo complejo f.
- Usa pesos de cercanía pi_x(z).
- Puede usar K-Lasso para seleccionar pocas variables.
- SP-LIME permite elegir explicaciones representativas para entender mejor el modelo.
SHAP:
- Explica una predicción individual.
- Es local.
- Se basa en Shapley Values.
- Reparte la predicción entre las variables de forma justa.
- Usa valores phi_i para indicar cuánto aporta cada variable.
- Cumple propiedades como local accuracy, missingness y consistency.
- Calcular SHAP exacto puede ser caro.
- Kernel SHAP aproxima los valores SHAP usando regresión ponderada.
Idea final para recordar
LIME = aproxima localmente un modelo complejo usando un modelo simple.
SHAP = reparte la predicción entre las variables usando contribuciones justas.
Más completo:

LIME perturba el dato, observa cómo cambia la predicción y entrena un explicador local.
SHAP calcula cuánto aporta cada variable a la predicción, considerando todas las coaliciones posibles.
