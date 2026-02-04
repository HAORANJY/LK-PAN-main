# A Large-Kernel Perceptual Attention Network for Robust Multi‑View Geo‑Localization with Drone‑View Offset Adaptation



## Abstract
Multi-view geo-localization utilizing drone and satellite imagery offers a reliable alternative to GPS-based positioning in challenging environments such as urban canyons and electromagnetically degraded areas. A key challenge in this task stems from the spatial misalignment between drone-view and satellite-view images, particularly when target buildings appear off-center due to varying flight attitudes and environmental disturbances. Existing methods, which often rely on implicit center-aligned assumptions, show limited robustness under such offset conditions. To address this limitation, we construct a novel drone-view offset dataset named Offset-1652 by applying controlled translational transformations to the benchmark dataset, simulating realistic displacement scenarios along horizontal, vertical, and diagonal directions. Furthermore, we propose a Large-Kernel Perceptual Attention Network (LK-PAN) that employs large-kernel depthwise convolutions to expand the receptive fields, thereby capturing global contextual information even for off-center targets. A symmetric InfoNCE loss is introduced to enhance cross-modal feature alignment and improve discrimination of hard negative samples. Comprehensive experiments conducted on both established benchmark datasets and the newly developed Offset-1652 and Offset-1652-MIX datasets demonstrate that the proposed method significantly outperforms existing approaches under a wide range of offset conditions. These results confirm its superior robustness and generalization capability for practical multi-view geo-localization in complex operational environments. The source code and datasets are available at: https://github.com/HAORANJY/LK-PAN-main.
## Dataset and download:
The training datasets utilized in this study are the publicly available SUES-200 [https://github.com/Reza-Zhu/SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark) and University-1652 [https://github.com/layumi/University1652-Baseline](https://github.com/layumi/University1652-Baseline). To generate the offset-augmented versions, the provided `Offset.py` script can be executed by specifying the source directory path (e.g., `/data/University-Release/train`), which produces 40 distinct offset variant files. The composite Offset-1652-MIX dataset is available for academic research upon request by contacting sunbo_bosun@chd.edu.cn or via [https://pan.quark.cn/s/c1f31c726021](https://pan.quark.cn/s/c1f31c726021) (password: JaT6). This benchmark collection will be progressively expanded in future releases.

## Requisites
- Python >= 3.8
- GPU Memory >= 6G
- Pytorch 1.13.0+cu116
- Torchvision 0.14.0+cu116


