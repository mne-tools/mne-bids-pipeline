# <img src="https://raw.github.com/mne-tools/mne-bids-pipeline/main/docs/source/assets/mne.svg" alt="Logotipo de MNE" height="20"> MNE-BIDS-Pipeline

<!-- hy-mt2-i18n:start -->
[English](./README.md) | [中文](./README_zh-CN.md) | [日本語](./README_ja.md) | **Español**
<!-- hy-mt2-i18n:end -->


<!--mantener la descripción en sincronía con pyproject.toml-->

<!--tagline-start-->
**MNE-BIDS-Pipeline es una pipeline de procesamiento completa para sus datos de MEG y EEG.**

* Funciona con datos almacenados según la [Estructura de Datos para Imágenes Cerebrales (BIDS)](https://bids.neuroimaging.io/).  
* En su interior, utiliza [MNE-Python](https://mne.tools).

<!--tagline-end-->

## 💡 Conceptos básicos y características

<!--features-list-start-->

* 🏆 Procesamiento automático de datos MEG y EEG, desde los datos brutos hasta las soluciones inversas.  
* 🛠️ Configuración mediante un sencillo archivo de texto.  
* 📘 Informes detallados de resumen sobre el procesamiento y análisis.  
* 🧑‍🤝‍🧑 Puede procesar a un único participante o hasta cientos de ellos en paralelo.  
* 💻 Ejecución a través de una utilidad de línea de comandos fácil de usar.  
* 🆘 Mensajes de error útiles en caso de que surjan problemas.  
* 👣 El procesamiento de datos se realiza como una secuencia de pasos estándar.  
* ⏩ Los pasos se almacenan en caché para evitar recálculos innecesarios.  
* ⏏️ Los datos pueden ser “expulsados” del pipeline en cualquier etapa. ¡Sin bloqueo!  
* ☁️ Funciona en su portátil, en un servidor potente o en un clúster de alto rendimiento mediante Dask.

<!--features-list-end-->

## 📘 Instrucciones de instalación y uso

Puede encontrar la documentación en
[**mne.tools/mne-bids-pipeline**](https://mne.tools/mne-bids-pipeline).

## ❤ Agradecimientos

El pipeline original para el procesamiento de datos MEG/EEG con MNE-Python fue creado conjuntamente por el [Cognition and Brain Dynamics Team](https://brainthemind.com/) y el [MNE Python Team](https://mne.tools), basado en scripts desarrollados originalmente para esta publicación:

> M. Jas, E. Larson, D. A. Engemann, J. Leppäkangas, S. Taulu, M. Hämäläinen,  
> A. Gramfort (2018). Un estudio grupal reproducible de MEG/EEG con el software MNE:  
> recomendaciones, evaluaciones de calidad y buenas prácticas. Frontiers in  
> neuroscience, 12. <https://doi.org/10.3389/fnins.2018.00530>

La iteración actual se basa en BIDS y depende de las extensiones de BIDS para EEG y MEG. Consulte las dos referencias siguientes:

> Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G.,
> Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS, una extensión
> de la estructura de datos para imágenes cerebrales destinada a la
> electroencefalografía. Scientific Data, 6, 103. <https://doi.org/10.1038/s41597-019-0104-8>

> Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A.,
> Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J.,
> Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, la estructura de datos
> para imágenes cerebrales extendida a la magnetoencefalografía. Scientific Data, 5, 180110.
> <https://doi.org/10.1038/sdata.2018.110>

## Contribuir

ver <./CONTRIBUTING.md>
