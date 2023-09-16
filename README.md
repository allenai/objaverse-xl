# 🪐 Objaverse-XL

This repository contains scripts to download and process Objaverse-XL.

<img src="https://mattdeitke.com/static/1cdcdb2ef7033e177ca9ae2975a9b451/9c1ca/objaverse-xl.webp">

Objaverse-XL is an open dataset of over 10 million 3D objects!

With it, we train Zero123-XL, a foundation model for 3D, observing incredible 3D generalization abilities: 🧵👇

## Scale Comparison

Objaverse 1.0 was released back in December. It was a step in the right direction, but still relatively small with 800K objects.

Objaverse-XL is over an order of magnitude larger and much more diverse!

<img src="https://github.com/allenai/objaverse-rendering/assets/28768645/43833dd3-ec97-4a3d-8782-00a6aea584b4">

## Unlocking Generalization

Compared to the original Zero123 model, Zero123-XL improves remarkably in 0-shot generalization abilities, even being able to perform novel view synthesis on sketches, cartoons, and people!

A ton more examples in the [📝 paper](https://arxiv.org/abs/2307.05663) :)

<img src="https://github.com/allenai/objaverse-rendering/assets/28768645/8470e4df-e39d-444b-9871-58fbee4b87fd">

## Image → 3D

With the base Zero123-XL foundation model, we can perform image → 3D using [DreamFusion](https://dreamfusion3d.github.io/), having the model guide a NeRF to generate novel views!

https://github.com/allenai/objaverse-rendering/assets/28768645/571852cd-dc02-46ce-b2bb-88f64a67d0ac

## Text → 3D

Text-to-3D comes for free with text → image models, such as with SDXL here, providing the initial image!

https://github.com/allenai/objaverse-rendering/assets/28768645/96255b42-8158-4c7a-8308-7b0f1257ada8

## Scaling Trends

Beyond that, we show strong scaling trends for both Zero123-XL and [PixelNeRF](https://alexyu.net/pixelnerf/)!

<img src="https://github.com/allenai/objaverse-rendering/assets/28768645/0c8bb433-27df-43a1-8cb8-1772007c0899">

## Tutorial

Check out the [Google Colab tutorial](//colab.research.google.com/drive/1zd4ri7ie_i5TYSUA9xHARh5W8nzrYpwg?usp=sharing) to download Objaverse-XL (work in progress).

## Blender Rendering

Blender rendering scripts are available in the [scripts/rendering directory](https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering)!

![266879371-69064f78-a752-40d6-bd36-ea7c15ffa1ec](https://github.com/allenai/objaverse-xl/assets/28768645/2f042d94-090b-4fd0-b37d-23b5971987ed)

## License

The use of the dataset as a whole is licensed under the ODC-By v1.0 license. Individual objects in Objaverse-XL are licensed under different licenses.

## Citation

To cite Objaverse-XL, please cite our [📝 arXiv](https://arxiv.org/abs/2307.05663) paper with the following BibTeX entry:

```bibtex
@article{objaverseXL,
  title={Objaverse-XL: A Universe of 10M+ 3D Objects},
  author={Matt Deitke and Ruoshi Liu and Matthew Wallingford and Huong Ngo and
          Oscar Michel and Aditya Kusupati and Alan Fan and Christian Laforte and
          Vikram Voleti and Samir Yitzhak Gadre and Eli VanderBilt and
          Aniruddha Kembhavi and Carl Vondrick and Georgia Gkioxari and
          Kiana Ehsani and Ludwig Schmidt and Ali Farhadi},
  journal={arXiv preprint arXiv:2307.05663},
  year={2023}
}
```

Objaverse 1.0 is available on 🤗Hugging Face at [@allenai/objaverse](https://huggingface.co/datasets/allenai/objaverse). To cite it, use:

```bibtex
@article{objaverse,
  title={Objaverse: A Universe of Annotated 3D Objects},
  author={Matt Deitke and Dustin Schwenk and Jordi Salvador and Luca Weihs and
          Oscar Michel and Eli VanderBilt and Ludwig Schmidt and
          Kiana Ehsani and Aniruddha Kembhavi and Ali Farhadi},
  journal={arXiv preprint arXiv:2212.08051},
  year={2022}
}
```

