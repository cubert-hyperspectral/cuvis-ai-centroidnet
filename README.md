# cuvis-ai-centroidnet

CentroidNetV2 for [Cuvis.AI](https://docs.cuvis.ai/latest/): count and separate many small objects that touch each other, on hyperspectral cubes.

One fully convolutional network predicts three things for every pixel: a vector to the centre of the nearest object, a vector to the border of that object, and class logits. The centre vectors are summed as votes. Peaks of the vote map are the objects, so their number is the count.

This plugin holds the parts of the method that are specific to it. The backbone is the `DynUNet` node of [cuvis-ai-unet](https://github.com/cubert-hyperspectral/cuvis-ai-unet), and the vote map is decoded by two builtin cuvis-ai nodes.

## Status

Version 0.1 is under construction. The nodes below exist; the target encoder, the vector loss, the detection metric and a training example follow.

| Node | What it does |
|---|---|
| `CentroidNetHeads` | Splits a backbone output with `4 + K` planes into centre vectors, border vectors and `K` class logits |
| `VotePeakCount` | Counts the vote peaks that reach a threshold, one number per image, without labels |

## Installation

The package is installed from a git tag once the first release is cut. Until then, from a checkout:

```bash
uv sync --extra dev
```

The count pipeline also needs `cuvis-ai-unet` and a cuvis-ai that ships `VectorFieldVoteAccumulator` and `LocalMaximaSuppression`.

## Plugin manifest

`plugins.yaml` at the repository root is a bare manifest with a local path source. A pipeline lists the plugin by name:

```yaml
plugins:
- unet
- centroidnet
```

`cuvis_ai_centroidnet/configs/pipeline/centroidnet_count.yaml` is a ready count pipeline: data node, per-frame standardisation, `DynUNet` with six output planes, `CentroidNetHeads`, vote accumulation, local-maxima suppression and `VotePeakCount`.

## Method and references

The nodes are written from the two papers below. Each class docstring names the section and equation it follows. The code contains no code of the authors' reference repositories.

- K. Dijkstra, J. van de Loosdrecht, W. A. Atsma, L. R. B. Schomaker, M. A. Wiering. "CentroidNetV2: A hybrid deep neural network for small-object segmentation and counting." Neurocomputing 423 (2021), pages 490 to 505. <https://doi.org/10.1016/j.neucom.2020.10.075>
- K. Dijkstra, J. van de Loosdrecht, L. R. B. Schomaker, M. A. Wiering. "CentroidNet: A deep neural network for joint object localization and counting." ECML PKDD 2018, LNAI 11053, pages 585 to 601, Springer, 2019. <https://doi.org/10.1007/978-3-030-10997-4_36>

If you use this plugin in research, please cite the papers:

```bibtex
@article{dijkstra2021centroidnetv2,
  author  = {Dijkstra, Klaas and van de Loosdrecht, Jaap and Atsma, Waatze A. and Schomaker, Lambert R. B. and Wiering, Marco A.},
  title   = {{CentroidNetV2}: A hybrid deep neural network for small-object segmentation and counting},
  journal = {Neurocomputing},
  volume  = {423},
  pages   = {490--505},
  year    = {2021},
  doi     = {10.1016/j.neucom.2020.10.075}
}

@inproceedings{dijkstra2019centroidnet,
  author    = {Dijkstra, Klaas and van de Loosdrecht, Jaap and Schomaker, Lambert R. B. and Wiering, Marco A.},
  title     = {{CentroidNet}: A deep neural network for joint object localization and counting},
  booktitle = {Machine Learning and Knowledge Discovery in Databases, ECML PKDD 2018},
  series    = {Lecture Notes in Computer Science},
  volume    = {11053},
  pages     = {585--601},
  publisher = {Springer},
  year      = {2019},
  doi       = {10.1007/978-3-030-10997-4_36}
}
```

## License

Apache-2.0, see [LICENSE](LICENSE).
