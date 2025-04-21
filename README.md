# Cityscape-Adverse 🌆

A robust benchmark for evaluating semantic segmentation models under adverse conditions using diffusion-based scene modifications.

## Overview

Cityscape-Adverse extends the original Cityscapes dataset by introducing realistic environmental variations generated through diffusion-based image editing. It provides a comprehensive benchmark for testing semantic segmentation model robustness under:

- 🌧️ Weather variations
- 💡 Lighting conditions
- 🍂 Seasonal changes

> This project is built on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) by OpenMMLab.

## Quick Start

### Prerequisites

1. **MMSegmentation Installation**
```bash
# Follow the detailed installation guide
pip install -v -e .
```
See [detailed installation instructions](docs/en/get_started.md#installation) for more options.

2. **Diffusion Tools Setup**
```bash
pip install -r requirements/imageedit.txt
```

### Dataset Access

#### Option 1: Download Pre-generated Dataset
Access our dataset through:
- 🤗 Huggingface: [naufalso/cityscape-adverse](https://huggingface.co/datasets/naufalso/cityscape-adverse)
- CLI Download:
```bash
python tools/editing/download_cityscape_adverse.py
```

#### Option 2: Generate Custom Data
> 📝 Detailed generation tutorial coming soon!

### Evaluation Guide
> 📝 Training and evaluation tutorial coming soon!

## Acknowledgements

This work builds upon:
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) by OpenMMLab

## Citation

If you use this work in your research, please cite:

```bibtex
@ARTICLE{10870179,
  author={Suryanto, Naufal and Adiputra, Andro Aprila and Kadiptya, Ahmada Yusril and Le, Thi-Thu-Huong and Pratama, Derry and Kim, Yongsu and Kim, Howon},
  journal={IEEE Access}, 
  title={Cityscape-Adverse: Benchmarking Robustness of Semantic Segmentation with Realistic Scene Modifications via Diffusion-Based Image Editing}, 
  year={2025},
  doi={10.1109/ACCESS.2025.3537981}
}
```

## License
This project is released under the [Apache 2.0 license](LICENSE).