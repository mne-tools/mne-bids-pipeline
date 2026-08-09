# <img src="https://raw.github.com/mne-tools/mne-bids-pipeline/main/docs/source/assets/mne.svg" alt="MNE Logo" height="20"> MNE-BIDS-Pipeline

<!-- hy-mt2-i18n:start -->
[English](./README.md) | [中文](./README_zh-CN.md) | **日本語** | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->


<!--pyproject.tomlの記述と整合性を保つ-->

<!--tagline-start-->
**MNE-BIDS-Pipelineは、MEGおよびEEGデータを対象とした機能十分な処理パイプラインです。**

* [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/) に従って保存されたデータを扱うことができます。
* 内部では [MNE-Python](https://mne.tools) を利用しています。

<!--tagline-end-->

## 💡 基本概念と機能

<!--features-list-start-->

* 🏆 生データから逆解析までのMEGおよびEEGデータの自動処理。  
* 🛠️ シンプルなテキストファイルを通じた設定。  
* 📘 詳細な処理結果および分析概要レポートの生成。  
* 🧑‍🤝‍🧑 単一の被験者のみならず、数百人規模の被験者も並列して処理可能。  
* 💻 使用しやすいコマンドラインユーティリティによる実行。  
* 🆘 何か問題が発生した際に役立つエラーメッセージの表示。  
* 👣 データ処理を一連の標準的なステップとして実施。  
* ⏩ 不要な再計算を避けるため、各ステップはキャッシュされる。  
* ⏏️ パイプライン内の任意の段階でデータを「除外」できる。ロックインの心配なし！  
* ☁️ ノートパソコン、高性能サーバー、またはDaskを利用した高性能クラスタ上で実行可能。

<!--features-list-end-->

## 📘 インストールおよび使用方法

ドキュメントは
[**mne.tools/mne-bids-pipeline**](https://mne.tools/mne-bids-pipeline) にあります。

## ❤ 致謝

MNE-Pythonを用いたMEG/EEGデータ処理の元のパイプラインは、この論文のために当初開発されたスクリプトを基に、
[Cognition and Brain Dynamics Team](https://brainthemind.com/)および[MNE Python Team](https://mne.tools)によって共同で構築されました。

> M. Jas, E. Larson, D. A. Engemann, J. Leppäkangas, S. Taulu, M. Hämäläinen, A. Gramfort (2018). MNEソフトウェアを用いた再現可能なMEG/EEG集団研究：推奨事項、品質評価、およびベストプラクティス。Frontiers in Neuroscience, 12. <https://doi.org/10.3389/fnins.2018.00530>

現在のバージョンはBIDSに基づいており、EEGおよびMEG向けのBIDS拡張機能を利用しています。以下の2つの参考文献をご覧ください：

> Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G., Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS、すなわち脳画像データ構造の電気脳波記録向け拡張版。Scientific Data, 6, 103. <https://doi.org/10.1038/s41597-019-0104-8>

> Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A., Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J., Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS、すなわち磁気脳波計測用に拡張された脳画像データ構造。Scientific Data, 5, 180110. <https://doi.org/10.1038/sdata.2018.110>

## 貢献方法

<./CONTRIBUTING.md> を参照してください。
