# Plan de trabajo — TP Final PAIByB
### Cariotipado automático de cromosomas
**Materia:** 16.85 Procesamiento Avanzado de Imágenes en Biología y Biomedicina · Grupo de 2 (Santiago + Kiwi)

---

## 1. Qué pide la consigna (checklist oficial)

El TP consiste en un algoritmo libre de procesamiento de imágenes en un contexto clínico/biológico, con **interfaz gráfica**, **informe de 6–8 páginas** y **defensa oral (~20 min)**. Es **obligatorio incluir TODAS las etapas** vistas en la materia:

| Etapa de la materia | ¿Obligatoria? | Cómo la cubrimos en cariotipado |
|---|---|---|
| **Contextualización** — obtención de imágenes | Sí | Dataset AutoKary2022 (imágenes de metafase, tinción Giemsa) |
| **Pre-procesamiento** — mejoramiento/denoising (Clase 1) | Sí | **FALTA** — hay que agregarlo explícitamente |
| **Extracción de características** (Clase 2) | Sí | PCA de eje principal + features de la CNN |
| **Segmentación** | Sí | Mask R-CNN (Detectron2) — casi listo |
| **Registración / fusión** (Clase 3) | Sí | Rectificación de cada cromosoma a orientación vertical canónica vía PCA |
| **Tracking de objetos/órganos** (Clase 4) | Sí | Punto débil — ver §4, requiere justificación o adaptación |
| **CNN / Machine Learning** (Clases 5–8) | Sí (fundamental) | Mask R-CNN (seg) + CNN de clasificación (VGG/ResNet) |
| **Reporte de resultados / diagnóstico** — visualización | Sí | Ensamblado del cariograma ordenado + interfaz |

**Estructura del informe exigida:** 1) Introducción (marco teórico, estado del arte, objetivos) · 2) Materiales y Métodos (incluye descripción de la interfaz) · 3) Resultados · 4) Conclusiones (comparación con estado del arte + mejoras).

> Nota: el archivo `Consigna_FINAL.pdf` original era un cuestionario de otra materia (Equipos Médicos). La consigna correcta es `Consigna.pdf`.

---

## 2. Estado actual del trabajo

### Segmentación (`Pipelines/Segmentación.ipynb`) — **~90% listo (Santiago)**
- Conversión LabelMe → COCO (train/test) ✔
- Registro del dataset en Detectron2 ✔
- Mask R-CNN ResNet-50 FPN, 1 clase ("chromosome"), entrada 1600×1600, ~50 epochs ✔
- Modelo entrenado guardado en `Modelos/Segmentacion/model_final.pth` (351 MB) ✔
- **Métricas (test):** AP box 81.5% · AP seg 78.9% · **AP50 97.8%** · AP75 94% ✔
- Extracción de cromosomas: máscara → crop → eje principal PCA → rotación a vertical → escalado/centrado a 224×224 con fondo blanco ✔

### Clasificación (`Pipelines/Clasificación.ipynb`) — **incompleto (Kiwi)**
- Extracción de cromosomas individuales a carpetas por clase (1–24) ✔
- Fine-tuning de ResNet50 y VGG16 (VGG dio mejor) ✔ pero:
  - **Falta reportar accuracy final y matriz de confusión limpia**
  - **Falta el modelo definitivo guardado** (no hay nada en `Modelos/Clasificacion/`)
  - Capa de atención mencionada pero no implementada
  - Grad-CAM presente pero desordenado ("HAY QUE VER COMO LABURAR CON EL HEATMAP")

### Interfaz (`Interfaz/`) — **~20% listo, mínima**
- Flask: subir imagen + botón "segmentar" (usa Detectron2, carga perezosa) ✔
- Solo visualiza segmentación; **no hay preprocesamiento, ni clasificación, ni cariograma final**
- Pesos del modelo no conectados por defecto (dependen de variable de entorno)

### `detectron2/` — clon completo del repo (14 MB)
- Es una **dependencia** para inferencia, no código nuestro. Ver §5.

---

## 3. Gap analysis — qué falta para cumplir la consigna

**Bloqueantes (obligatorios por consigna):**
1. **Pre-procesamiento explícito** (Clase 1): estimación de ruido + denoising + realce de contraste/normalización. Hoy no existe como etapa.
2. **Registración** (Clase 3): formalizar la rectificación PCA como "registración a plantilla canónica" (ya está hecha técnicamente, falta encuadrarla y documentarla).
3. **Tracking** (Clase 4): etapa que peor encaja en imágenes estáticas → decidir enfoque (§4).
4. **Terminar clasificación**: modelo final + métricas + guardado en `Modelos/`.
5. **Ensamblado del cariograma**: ordenar los cromosomas clasificados (1–22 + XY) en la grilla estándar = salida final / "diagnóstico".
6. **Interfaz completa**: pipeline end-to-end con visualización paso a paso.
7. **Informe 6–8 páginas** (no iniciado).

**Riesgo técnico transversal:** segmentación en **PyTorch/Detectron2** y clasificación en **TensorFlow/Keras**. Conviven en un mismo entorno pero es pesado; hay que unificar el entorno de la interfaz (§5).

---

## 4. Estructura propuesta del pipeline (end-to-end)

```
IMAGEN MICROSCÓPICA (metafase)
        │
   [1] PRE-PROCESAMIENTO           ← Clase 1 (denoising) + Clase 2
        │  · estimación de ruido (p.ej. MAD sobre wavelets / varianza en zonas planas)
        │  · denoising (Non-Local Means / filtro bilateral / mediana)
        │  · normalización de intensidad + realce de contraste (CLAHE)
        ▼
   [2] SEGMENTACIÓN                 ← Mask R-CNN (ya entrenado)
        │  · máscaras de instancia por cromosoma
        ▼
   [3] EXTRACCIÓN + REGISTRACIÓN    ← Clase 2 + Clase 3
        │  · recorte por máscara
        │  · eje principal (PCA) → rotación a vertical (registración a plantilla)
        │  · escalado relativo + centrado a 224×224
        ▼
   [4] CLASIFICACIÓN                ← CNN (VGG/ResNet + atención)
        │  · etiqueta 1–22 + X/Y por cromosoma
        ▼
   [5] ENSAMBLADO DEL CARIOGRAMA    ← Reporte/diagnóstico
           · grilla ordenada + conteo por par
           · flag de posibles anomalías numéricas (p.ej. ≠2 por clase)
```

**Sobre "Tracking" (Clase 4) — 3 opciones, elegir una:**
- **(A) Justificar la no-aplicabilidad** en el informe: el cariotipado opera sobre imágenes estáticas; no hay dimensión temporal. Se menciona la etapa y por qué no aplica. *(Más honesto, pero la consigna dice "obligatorio incluir cada paso".)*
- **(B) Reencuadrar como "asociación/seguimiento de instancias"**: matching de cada instancia segmentada entre la imagen microscópica y su posición en el cariograma final (tracking espacial de correspondencias). *(Recomendada — cumple formalmente y es coherente.)*
- **(C) Demo separada de tracking real** sobre una secuencia de división celular (time-lapse de mitosis) como anexo, para exhibir la técnica aunque no sea parte del flujo principal.

→ **Recomendación: (B) como parte del flujo + una línea de (A) en el informe.** Consultar al docente Roberto Tomás si hay dudas.

---

## 5. Interfaz — decisión tecnológica

**Recomendación: migrar de Flask a Streamlit (o Gradio).**

| Criterio | Flask (actual) | **Streamlit** ⭐ | Gradio |
|---|---|---|---|
| Velocidad de desarrollo | Baja (rutas, HTML, JS) | **Muy alta** | Alta |
| Visualización paso a paso | Manual | **Nativa (columnas, imágenes, sliders)** | Buena |
| Ideal para demo ML académica | Medio | **Sí** | Sí |
| Control fino / producción | Alto | Medio | Bajo |

La consigna pide "ejecutar los algoritmos **y visualizar el procesamiento realizado**". Streamlit permite mostrar cada etapa (original → denoised → máscaras → cromosomas rectificados → cariograma) con muy poco código. Flask obliga a escribir plantillas y manejo de estado a mano.

**Entorno único de la interfaz** (resuelve el problema PyTorch vs TensorFlow):
- Un solo venv con `torch` + `detectron2` (segmentación) **y** `tensorflow` (clasificación). Funciona, es pesado (~varios GB).
- Alternativa más limpia a futuro: reexportar el clasificador a PyTorch/ONNX y quedarse con un solo framework. *(No es prioritario ahora.)*

**Qué hacer con `detectron2/`:**
- No es código del grupo: es dependencia. **Instalarlo por `pip`/requirements en el venv de la interfaz** en lugar de versionar el clon completo.
- Sacar la carpeta del entregable y del informe (o `.gitignore`). Mantenerla solo si necesitan instalación offline reproducible.
- **No** hace falta para "visualizar pasos" por sí misma; la visualización la damos nosotros con matplotlib/OpenCV sobre las salidas del modelo.

---

## 6. Plan de acción por fases

### Fase 1 — Cerrar lo técnico (núcleo del TP)
1. **Santiago — Pre-procesamiento** (bloqueante): notebook/módulo `preprocesamiento` con estimación de ruido, denoising (NLM/bilateral), CLAHE + normalización, y comparación cuantitativa (PSNR/SSIM antes-después). *Cubre Clase 1 y 2.*
2. **Kiwi — Terminar clasificación**: elegir modelo final (VGG + atención), reentrenar, reportar accuracy + matriz de confusión + Grad-CAM limpio, y **guardar modelo** en `Modelos/Clasificacion/`.
3. **Ambos — Ensamblado del cariograma**: función que ordena cromosomas clasificados en la grilla estándar y cuenta pares (salida final + flags de anomalía).
4. **Formalizar registración y tracking** (opción 4B) en un módulo/documentación.

### Fase 2 — Integración
5. **Pipeline unificado** `preproc → seg → extracción/registración → clasif → cariograma` en un solo script/módulo reutilizable por la interfaz.
6. **Interfaz Streamlit** que ejecute el pipeline y muestre cada etapa; conectar pesos de ambos modelos; limpiar dependencia detectron2.

### Fase 3 — Entregables
7. **Informe 6–8 páginas** (LaTeX recomendado) siguiendo la estructura exigida; comparar métricas con el estado del arte (paper Wang 2024: seg AP50 96.6%, clasif 95.2% / AutoKary2022: mAP@50 multiclase 65.9%).
8. **Preparar defensa oral (~20 min)**: demo en vivo de la interfaz + resultados.
9. **Verificación final**: correr el pipeline end-to-end sobre imágenes de test, chequear métricas y que la interfaz funcione de punta a punta.

---

## 7. Estructura de carpetas sugerida

```
Final PAIByB/
├── Consigna.pdf
├── Datasets/Autokary2022_1600x1600/
├── Pipelines/
│   ├── 1_Preprocesamiento.ipynb      ← NUEVO
│   ├── 2_Segmentación.ipynb          ← existe
│   ├── 3_Clasificación.ipynb         ← terminar
│   └── 4_Pipeline_Integrado.ipynb    ← NUEVO
├── Modelos/
│   ├── Segmentacion/model_final.pth  ← existe
│   └── Clasificacion/                ← FALTA modelo final
├── Interfaz/                          ← migrar a Streamlit
├── Informe/                          ← NUEVO (LaTeX)
└── Papers/
```

---

## 8. Decisiones abiertas (a definir con el grupo / docente)
- Enfoque de **Tracking** (4A / 4B / 4C).
- **Interfaz**: confirmar Streamlit vs seguir con Flask.
- Alcance del **preprocesamiento** (qué técnicas y cómo evaluarlas).
- ¿Se elimina el clon de `detectron2/` del repo?
