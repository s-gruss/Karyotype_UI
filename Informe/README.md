# Informe final — Cariotipado automático

Informe en LaTeX del TP Final (PAIByB, 16.85).

## Archivos
- `informe.tex` — documento principal.
- `referencias.bib` — bibliografía (BibTeX, estilo IEEEtran).
- `figuras/` — carpeta para las imágenes (crear y completar).

> El informe no referencia las "Clases" de la asignatura en ningún título ni texto.

## Compilar (TeXstudio)
Configurar la secuencia **pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX**
(o usar el atajo de compilación completa, F5). Desde consola:

```bash
pdflatex informe
bibtex   informe
pdflatex informe
pdflatex informe
```

## Figuras pendientes
El `.tex` compila sin imágenes: cada figura usa un placeholder (`\figplaceholder`).
Para reemplazarlas, poner la imagen en `figuras/` y cambiar la línea
`\figplaceholder{...}` por `\includegraphics[width=...]{figuras/nombre}`.

Figuras a generar:
1. `fig:pipeline` — esquema del pipeline.
2. `fig:interfaz` — captura de la interfaz Streamlit.
3. `fig:seg-ejemplo` — imagen de metafase + máscaras predichas.

## Pendientes de redacción
- **Resumen**: se dejó como placeholder (después de la carátula, antes del índice);
  se redacta al finalizar el trabajo.
- Verificar los valores de PSNR/SSIM del pre-procesamiento contra el notebook
  `1_Preprocesamiento.ipynb` y ajustarlos si difieren.
- La etapa de clasificación / cariograma se dejó fuera del informe (aún no
  definida); figura como trabajo futuro en las conclusiones.

## Estructura
1. Carátula (página completa).
2. Resumen (pendiente) + Índice.
3. Introducción · Materiales y Métodos · Resultados · Conclusiones · Bibliografía.
