# Informe final — Cariotipado automático

Informe en LaTeX del TP Final (PAIByB, 16.85).

## Archivos
- `informe_final.tex` — **documento principal** (único informe canónico).
- `referencias.bib` — bibliografía (BibTeX, estilo IEEEtran).
- `figuras/` — imágenes (curvas de entrenamiento de la clasificación, etc.).

> El informe no referencia las "Clases" de la asignatura en ningún título ni texto,
> ni usa `\emph` ni aclaraciones entre guiones largos.

## Compilar (TeXstudio)
Configurar la secuencia **pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX**
(o usar el atajo de compilación completa, F5). Desde consola:

```bash
pdflatex informe_final
bibtex   informe_final
pdflatex informe_final
pdflatex informe_final
```

## Figuras pendientes
Las figuras de clasificación (curvas de entrenamiento) ya están incluidas desde
`figuras/`. Quedan como placeholder (`\figplaceholder`) tres figuras a generar:
1. `fig:pipeline` — esquema del pipeline.
2. `fig:interfaz` — captura de la interfaz Streamlit.
3. `fig:seg-ejemplo` — imagen de metafase + máscaras predichas.

Para reemplazarlas: poner la imagen en `figuras/` y cambiar `\figplaceholder{...}`
por `\includegraphics[width=...]{nombre}`.

## Pendientes de redacción
- **Resumen**: placeholder (después de la carátula, antes del índice); se redacta al
  final.
- **Cariograma**: ensamblado pendiente (marcado con `\pendiente{}` en Métodos y
  Resultados).
- Verificar los valores de PSNR/SSIM del preprocesamiento contra el notebook
  `1_Preprocesamiento.ipynb` y ajustarlos si difieren.

## Estado del preprocesamiento (importante)
Los modelos (segmentación y clasificación) se ejecutan sobre la **imagen cruda**, por
consistencia entrenamiento/inferencia. El preprocesamiento se presenta como etapa de
**caracterización** (estimación de ruido, denoising, CLAHE, medido con PSNR/SSIM) y
queda como herramienta de visualización en la interfaz, fuera de la cadena de
inferencia.

## Estructura
1. Carátula (página completa).
2. Resumen (pendiente) + Índice.
3. Introducción · Materiales y Métodos · Resultados · Conclusiones · Anexo (detalle
   de clasificación) · Bibliografía.
