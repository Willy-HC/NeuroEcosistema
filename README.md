# NeuroEcosistema

## Sonda de orientación neuronal hacia alimento

Ejecuta una prueba determinista de orientación de los modelos de presa 5G y
30G con ocho direcciones de planta y un escenario sin planta:

```powershell
uv run python -m neuroecosistema.analysis.food_orientation_probe `
  --output-dir "C:\ruta\de\salida"
```

Por defecto, la herramienta carga los archivos `best_prey.npz` de
`results/coevolution_diagnostics_5gen` y
`results/coevolution_diagnostics_30gen`. Escribe
`food_orientation_diagnostics.csv` y `food_orientation_summary.csv` únicamente
en el directorio indicado. Esta sonda es determinista y no constituye una
evaluación estadística de generalización.

Una salida se marca saturada cuando el valor absoluto de cualquiera de sus
logits previos a `tanh` es al menos 3 (equivalente a una magnitud de `tanh`
aproximadamente igual o superior a 0.995).