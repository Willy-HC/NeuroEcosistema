# NeuroEcosistema

## Sonda de orientación neuronal hacia alimento

Ejecuta una prueba determinista de orientación de las presas con tres
posiciones iniciales, distancias de alimento de 50, 100 y 200 unidades, ocho
direcciones y escenarios sin planta. Incluye la heurística como control
positivo:

```powershell
uv run python -m neuroecosistema.analysis.food_orientation_probe `
  --output-dir "C:\ruta\de\salida"
```

Por defecto, la herramienta carga los archivos `best_prey.npz` de
`results/coevolution_diagnostics_5gen` y
`results/coevolution_diagnostics_30gen`. Escribe 225 filas de diagnóstico y un
resumen por política, distancia, posición y eje horizontal/vertical únicamente
en el directorio indicado. Esta sonda es determinista y no constituye una
evaluación estadística de generalización.

Una salida se marca saturada cuando el valor absoluto de cualquiera de sus
logits previos a `tanh` es al menos 3 (equivalente a una magnitud de `tanh`
aproximadamente igual o superior a 0.995).

## Diagnóstico de condiciones de población

La sonda de condiciones compara los modelos congelados de presa 5G y 30G en
episodios de 500 pasos con poblaciones de 10 presas/10 depredadores,
50 presas/10 depredadores y 10 presas/0 depredadores:

```powershell
uv run python -m neuroecosistema.analysis.population_condition_probe `
  --output-dir "C:\ruta\de\salida" `
  --seeds 42 123 456 `
  --max-steps 500
```

En condiciones con depredadores cruza ambos modelos de presa con las políticas
de depredador 5G y 30G para no confundir sus efectos. Cada población está
formada por clones del mismo genoma; no es una repetición de la selección
coevolutiva original. Para reducir diferencias ambientales, fija por semilla
las plantas iniciales y las posiciones de los primeros 10 agentes de cada
especie en todas las condiciones. Los eventos de reposición posteriores pueden
divergir si las poblaciones consumen cantidades distintas.

La herramienta escribe `agent_diagnostics.csv`, `episode_summary.csv` y
`manifest.json` solamente en el directorio indicado. El manifiesto registra los
hashes de los modelos, la configuración efectiva, las posiciones iniciales y
las limitaciones de comparabilidad. Se pueden incluir semillas adicionales,
por ejemplo `--seeds 42 123 456 789`.

## Fitness experimental de progreso alimentario

El entrenamiento mantiene `baseline` como modo predeterminado. Para una
ejecución futura experimental de presas, se puede seleccionar
`--fitness-mode food_progress` y configurar `--food-progress-weight` (por
defecto `0.05`). La fórmula añade `peso * food_progress_score` al fitness
original; la alimentación, supervivencia y exploración conservan sus términos.
El modo experimental no se admite para entrenamiento solo de depredadores.

Cada presa mantiene referencias separadas para las identidades de plantas
activas. En cada movimiento se acumula el cambio firmado de distancia a cada
planta, normalizado por la diagonal del mapa. Una referencia solo produce
crédito si esa misma presa consume la planta; el crédito es el progreso neto
desde que se registró la planta hasta la colisión, limitado a `[0, 1]`. Las
plantas desaparecidas por consumo ajeno descartan solo su propia referencia,
y las reposiciones comienzan con progreso cero. El score es la media de los
créditos realizados; sin consumos es cero. El fitness experimental se limita
a `[0, 1]`. Esto evita acumular progreso de oscilaciones sin consumo, aunque
el progreso pendiente de una planta que consume otra presa no se acredita.
Depredadores y fitness `baseline` no reciben esta señal. Los metadatos de cada
modelo guardan el modo y el peso efectivo.

## Preentrenamiento por imitacion y coevolucion

El preentrenamiento sintetiza siete tipos de escenario comunes y usa un
curriculo de dos etapas: primero busqueda de alimento/persecucion y luego
amenazas, evasion y combinaciones, conservando ejemplos de busqueda en la
segunda etapa. Solo el curriculo de presas anade el escenario
`food_with_distant_threat`, que coloca depredadores fuera del radio de evasion
para reforzar que una amenaza lejana no debe desplazar la busqueda de alimento.
Entrenamiento, validacion y prueba usan semillas distintas y estados
sinteticos independientes que cubren las mismas familias de escena;
validacion selecciona el checkpoint y prueba se consulta despues. Los casos
sin objetivo no inventan una etiqueta direccional ni contribuyen al gradiente.
Cuando hay comida y depredadores, la etiqueta combina la busqueda con una
huida reactiva ponderada por la proximidad de la amenaza; sin comida, una
presa solo recibe etiqueta de movimiento si una amenaza cercana exige huir.
La perdida optimiza direccion angular (la misma que consume el controlador,
que normaliza las salidas) y penaliza salidas de magnitud casi nula. La segunda
etapa conserva por repeticion la mitad de sus ejemplos de busqueda/persecucion
para reducir el olvido al aprender evasion. Las metricas tambien informan
error angular, saturacion y sesgo direccional.
Los modelos `.npz` siguen siendo compatibles con el cargador existente.
Ejecuta cada etapa en un directorio de salida nuevo para evitar sobrescribir
resultados:

```powershell
uv run python -m neuroecosistema.pretrain_imitation `
  --output-dir models\imitation `
  --seed 8121901
```

Entrena ambas especies mediante coevolucion, sembrando cada poblacion con el
modelo correspondiente. Se conserva una fraccion aleatoria para diversidad;
los rivales historicos se habilitan por defecto:

```powershell
uv run python -m neuroecosistema.train `
  --species both `
  --generations 30 `
  --population-size 30 `
  --max-steps 500 `
  --pretrained-prey models\imitation\best_prey.npz `
  --pretrained-predator models\imitation\best_predator.npz `
  --pretrained-fraction 0.5 `
  --pretrained-noise 0.02 `
  --output results\imitation_coevolution
```

Compara politicas sobre semillas y escenarios reservados. La evaluacion escribe
resultados por episodio, resumenes y un manifiesto con parametros y hashes:

```powershell
uv run python -m neuroecosistema.evaluate_learning `
  --pretrained-model-dir models\imitation `
  --coevolved-model-dir results\imitation_coevolution `
  --output-dir results\imitation_evaluation `
  --seeds 789 1013 2027 `
  --scenario 10,2,30 `
  --scenario 30,5,60 `
  --scenario 50,10,100 `
  --max-steps 500
```

Para cargar los modelos coevolucionados en la simulacion visual:

```powershell
uv run python -m neuroecosistema.main `
  --trained `
  --model-dir results\imitation_coevolution
```

### Alcance y resultados disponibles

El profesor de presa combina la heuristica alimentaria existente con una
respuesta reactiva a amenazas; el de depredador usa la heuristica de
persecucion existente. La percepcion no incluye velocidades de objetivos, por
lo que la imitacion no aprende anticipacion. La evaluacion es finita y
reproducible, pero no garantiza generalizacion fuera de los escenarios
muestreados.

En un diagnostico acotado se entrenaron ambas redes con 4.000/700/700
escenarios sinteticos y hasta 30 epocas; despues se evaluaron 500 pasos con
tres semillas nuevas (3001, 3002 y 3003), fijando la politica de la especie
oponente como aleatoria. En 10/2/30, la presa neuronal consumio 0,97 plantas
por presa frente a 0,53 aleatorias, y el depredador capturo 0,50 presas frente
a 0,33 aleatorias. En 50/10/100, la presa consumio 1,89 frente a 2,05, y el
depredador capturo 2,47 frente a 1,23. Por tanto, la presa supera el control
aleatorio en la condicion de baja densidad, pero no en la alta; el criterio
alimentario general sigue incumplido y no se debe considerar lista para
coevolucion. El depredador supera el control aleatorio en ambas condiciones.

El error angular de entrenamiento/validacion/prueba fue 32,89/32,75/37,23
grados para presas y 31,12/32,54/32,27 para depredadores. No se midio
saturacion ni norma casi nula en las salidas etiquetadas. Esto no demuestra
que la conducta sea suficiente en el entorno: la evaluacion de consumo de la
presa aun falla en una condicion no vista. No se ejecuto coevolucion ni
preentrenamiento largo.

Como intervencion acotada se reforzo solo el curriculo de presas con escenas
de alimento visible y depredadores fuera del radio de evasion; el curriculo de
depredadores no cambia. Una candidata temporal entrenada con 4.000/700/700
escenas y hasta 30 epocas obtuvo 26,65 grados de error en esa familia de
prueba sintetica, pero no mejoro el consumo en episodios de 500 pasos con tres
semillas nuevas. En 10/2/30 consumio 0,53 plantas por presa frente a 0,87 del
control aleatorio; en 50/10/100 consumio 1,41 frente a 1,99. La supervivencia
media fue 500 frente a 500 pasos en baja densidad y 431 frente a 424 en alta.
No se cumplen los criterios de aceptacion: la candidata no se promovio y los
modelos existentes se conservaron. No se ejecuto coevolucion.

### Piloto DAgger para presas

`neuroecosistema.dagger_prey` parte de `models/imitation/best_prey.npz`, recoge
en hasta tres rondas las percepciones exactas de presas movidas por la red y
las etiquetas del controlador heuristico en episodios cortos con depredadores
aleatorios. Los estados sin planta objetivo se excluyen. Cada actualizacion
conserva los datos DAgger de rondas anteriores y una pequena memoria de los
escenarios supervisados existentes; usa el mismo optimizador supervisado.
Cada candidato se compara en las mismas semillas reservadas de 10/2/30 y
50/10/100 con la red inicial, el control aleatorio y el heuristico. La
seleccion considera conjuntamente alimento y retencion de supervivencia.
La salida se guarda en un directorio nuevo bajo `results/`, sin sobrescribir
el modelo inicial ni alterar el de depredador:

```powershell
uv run python -m neuroecosistema.dagger_prey `
  --initial-model models\imitation\best_prey.npz `
  --output-dir results\prey_dagger_pilot
```

En la corrida piloto de tres rondas (100 pasos de recoleccion; 500 de
evaluacion; semillas reservadas 8401201-8401203), la ronda 2 fue el mejor
balance: obtuvo 1,70 plantas por presa en 10/2/30 frente a 0,83 aleatorias y
2,65 en 50/10/100 frente a 1,87 aleatorias. Supero al aleatorio en las tres
semillas de ambos escenarios. La supervivencia fue 484,4/500 pasos en baja y
411,8/500 en alta densidad; en alta, esto supera los 405,2 pasos aleatorios,
pero es 10,3% inferior a la red inicial (459,1). Por el margen conservador
del piloto (90% de la supervivencia inicial), el candidato no se promueve a
`models/imitation/`; se conserva aparte en
`results/prey_dagger_pilot/best_prey.npz` para revision. La red original y el
modelo de depredador no se modificaron.