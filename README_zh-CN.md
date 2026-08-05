# <img src="https://raw.github.com/mne-tools/mne-bids-pipeline/main/docs/source/assets/mne.svg" alt="MNE Logo" height="20"> MNE-BIDS-Pipeline

<!-- hy-mt2-i18n:start -->
[English](./README.md) | **中文** | [日本語](./README_ja.md) | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->


<!--需使说明内容与pyproject.toml保持一致-->

<!--tagline-start-->
**MNE-BIDS-Pipeline 是一款专为处理您的脑磁图与脑电图数据而设计的功能完备的流程工具。**

* 它能够处理按照[脑成像数据结构（BIDS）](https://bids.neuroimaging.io/)格式存储的数据。  
* 在底层实现上，它基于[MNE-Python](https://mne.tools)开发。

<!--tagline-end-->

## 💡 基本概念与功能特性

<!--features-list-start-->

* 🏆 自动完成从原始数据到逆向解的MEG与EEG数据处理流程。  
* 🛠️ 通过简单的文本文件进行配置。  
* 📘 提供详尽的处理与分析汇总报告。  
* 🧑‍🤝‍🧑 可以单独处理一名参与者，也可并行处理数百名参与者。  
* 💻 通过易于使用的命令行工具来执行操作。  
* 🆘 出现问题时会有实用的错误提示。  
* 👣 将数据处理视为一系列标准步骤来完成。  
* ⏩ 会缓存各步骤结果以避免不必要的重复计算。  
* ⏏️ 可以在任意阶段将数据从处理流程中移除。无需担心数据锁定问题！  
* ☁️ 支持在笔记本电脑、高性能服务器或通过Dask运行的高性能集群上使用。

<!--features-list-end-->

## 📘 安装与使用指南

相关文档可在
[**mne.tools/mne-bids-pipeline**](https://mne.tools/mne-bids-pipeline) 查阅。

## ❤ 致谢

基于为该论文最初开发的脚本，MNE-Python用于MEG/EEG数据处理的原始处理流程是由[Cognition and Brain Dynamics Team](https://brainthemind.com/)与[MNE Python Team](https://mne.tools)共同构建的。

> M. Jas, E. Larson, D. A. Engemann, J. Leppäkangas, S. Taulu, M. Hämäläinen, A. Gramfort (2018). 利用MNE软件开展可重复的MEG/EEG群体研究：建议、质量评估与最佳实践。《神经科学前沿》，12卷。<https://doi.org/10.3389/fnins.2018.00530>

当前版本基于BIDS构建，并依赖于针对EEG和MEG的BIDS扩展。相关内容可参见以下两篇参考文献：

> Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G.,
> Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS，一种针对脑电图的脑成像数据结构扩展方案。《科学数据》，6，103。<https://doi.org/10.1038/s41597-019-0104-8>

> Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A.,
> Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J.,
> Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS，一种扩展用于脑磁图研究的脑成像数据结构。《科学数据》，5，180110。<https://doi.org/10.1038/sdata.2018.110>

## 贡献指南

参见 <./CONTRIBUTING.md>
