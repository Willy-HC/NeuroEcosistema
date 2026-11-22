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