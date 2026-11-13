# cuvis-ai-centroidnet

CentroidNetV2 for [Cuvis.AI](https://docs.cuvis.ai/latest/): find and count many small objects on hyperspectral cubes, also where they touch each other.

One fully convolutional network predicts three things for every pixel: a vector to the centre of the nearest object, a vector to the border of that object, and class logits. The centre vectors are summed as votes. Peaks of the vote map are the objects, so their number is the count.

This plugin holds the parts of the method that are specific to it. The backbone is the `DynUNet` node of [cuvis-ai-unet](https://github.com/cubert-hyperspectral/cuvis-ai-unet), and the vote map is decoded by two builtin cuvis-ai nodes.

## Nodes

| Node | What it does |
|---|---|
| `CentroidNetHeads` | Splits a backbone output with `4 + K` planes into centre vectors, border vectors and `K` class logits |
| `ConnectedComponentInstances` | Turns a class mask into an instance map, for annotations without object ids |
| `InstanceVectorFieldEncoder` | Turns an instance map into the training targets: both vector fields, an object mask and loss weights |
| `VectorFieldLoss` | Mean Euclidean distance between predicted and target vectors, in units of the vote range |
| `VotePeakCount` | Counts the vote peaks that reach a threshold, one number per image, without labels |
| `CentroidDetectionMetrics` | Precision, recall, F1 and count error of the vote peaks against the instance centroids, for several thresholds at once |

Two pipeline configs ship with the package, in `cuvis_ai_centroidnet/configs/pipeline/`:

- `centroidnet_fod_lentils_train.yaml` is the full graph: data, normalisation, targets, backbone, three losses, voting, metrics and count. The nodes that need labels run only in training, validation and test, so a pipeline trained with this config counts objects in a bare cube at inference.
- `centroidnet_count.yaml` is the label-free part of that graph with the same node names, for inference only.

What version 0.1 does not do: it gives object centres and counts, not instance masks. Fitting a shape through the border points of each centre is the next step of the method and is not implemented yet. [docs/implementation.md](docs/implementation.md) lists every part of the paper, the class that implements it and each point where the plugin differs.

## Installation

From a checkout:

```bash
uv sync --extra dev
```

The pipelines also need three sibling packages. Their versions are pinned in the `chain` job of `.github/workflows/ci.yml`; install them on top, without their dependencies:

```bash
uv pip install --no-deps \
  "cuvis-ai-unet @ git+https://github.com/cubert-hyperspectral/cuvis-ai-unet.git@v0.5.0" \
  "cuvis-ai-augment @ git+https://github.com/cubert-hyperspectral/cuvis-ai-augment.git@v0.5.0" \
  "cuvis-ai @ git+https://github.com/cubert-hyperspectral/cuvis-ai.git@main"
```

cuvis-ai has to be a version that ships `VectorFieldVoteAccumulator` and `LocalMaximaSuppression`, which is later than 0.22.8. After this step run commands with `uv run --no-sync`, because a plain `uv run` puts the released cuvis-ai back.

## Plugin manifest

`plugins.yaml` at the repository root is a bare manifest with a local path source. A pipeline lists the plugins it uses by name:

```yaml
plugins:
- unet
- augment
- centroidnet
```

## Example: foreign objects on lentils

`examples/fod_lentils/` trains the pipeline on the public `Industrial_FOD_Lentils` set (1136 frames of 1000 x 1080 pixels and 61 bands, 53 GiB) or on the 69-frame `Demo_Industrial_FOD_Lentils`. Download either with `cuvis_ai_core.data.public_datasets.PublicDatasets`. The scripts read the frames from cu3s recordings, which needs the [Cuvis SDK](https://www.cubert-hyperspectral.com/) and the dataloader plugin:

```bash
uv pip install "cuvis-ai-dataloader[cu3s,coco]>=0.9.1" "click>=8.1"
```

Train, with `<lentils>` the folder of the download:

```bash
cd examples/fod_lentils
uv run --no-sync python train.py --universe-csv <lentils>/universe.csv --splits-json <lentils>/splits/adaclip.json --out runs/lentils
```

The frames come straight from the cu3s recordings; there is no converted copy of the data. Reading a frame takes longer than a training step, so the scripts read on threads: `--read-threads 4` reads the frames of a batch at once, and `--read-ahead 4` reads the next ones while the model works. Only `--max-open-sessions` recordings stay open and opening one takes seconds, so every batch comes from one recording (`--coherent-batches`) and an epoch goes through the recordings one after the other, in a new order each epoch. With batches of 4 on an RTX 5070 Ti laptop GPU the training loader alone reads about 10 frames per second this way, 5 with one thread, and 1 with batches mixed from all recordings (`--mixed-batches`).

`--out` has to be a new folder for every run: the script refuses a folder that already holds a trained pipeline, so the threshold and scores of one model never end up next to the weights of another.

After the fit the framework scores the validation and the test split. The script takes the vote threshold with the highest F1 on validation, stores it with the pipeline (`pipeline.threshold.json` and the `Count` node of `pipeline.yaml`) and writes the test scores at that threshold to `pipeline.test_metrics.json`. `select_threshold.py` and `evaluate.py` repeat these two steps for a saved pipeline.

A full-frame pass runs the backbone on overlapping tiles, so scoring a split takes a GPU. Results of a full training run are not part of this version yet.

## Tests

```bash
uv run --no-sync pytest tests/ -m "not chain"   # the nodes on their own
uv run --no-sync pytest tests/ -m chain         # with DynUNet, the crop node and the vote nodes
```

The chain tests need the three sibling packages from the Installation section. With the dev extra alone they fail at import: the cuvis-ai release it installs (0.22.8) does not have the two vote nodes yet.

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
