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