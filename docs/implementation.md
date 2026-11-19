# How the paper maps to this plugin

This page lists, for every part of CentroidNetV2, where the paper describes it, which class
implements it, and where the implementation differs from the paper and why.

The method is from:

> K. Dijkstra, J. van de Loosdrecht, W. A. Atsma, L. R. B. Schomaker, M. A. Wiering.
> "CentroidNetV2: A hybrid deep neural network for small-object segmentation and counting."
> Neurocomputing 423 (2021), pages 490 to 505. <https://doi.org/10.1016/j.neucom.2020.10.075>

Section, equation and page numbers below refer to that paper. Its predecessor is CentroidNet
(Dijkstra, van de Loosdrecht, Schomaker, Wiering, ECML PKDD 2018,
<https://doi.org/10.1007/978-3-030-10997-4_36>).

## Origin of the code

The code is written from the paper. The authors' reference implementation,
[OpenCentroidNetV2](https://github.com/kdijkstra13/OpenCentroidNetV2) (MIT licence, read at
commit `ee01358`), was consulted for values the paper leaves open and to check that nothing
was missed. No code was taken from it, and none from the CentroidNet V1 repository.

## One graph of small nodes

The paper describes a backbone `f`, a decoder `g`, an encoder `g'` and a loss `l` (Fig. 1).
In Cuvis.AI each of these is one or more nodes, wired in a pipeline config:

| Paper | Where in the paper | Node | Package |
|---|---|---|---|
| Backbone `f`, `Y = f(X)` | Section 3.1, Eq. 1, page 492 | `DynUNet` with `num_classes = 4 + K` | cuvis-ai-unet |
| Output planes `Y = [Yc, Yb, Yl]` | Section 3.1, Eq. 2 | `CentroidNetHeads` | this plugin |
| Encoder `g'`: targets `Tc`, `Tb`, `Tp` | Section 3.3.2, pages 495 to 496 | `InstanceVectorFieldEncoder` | this plugin |
| Vector loss | Section 3.2.2, Eq. 6 and 7, page 493 | `VectorFieldLoss`, once per vector field | this plugin |
| Segmentation loss | Section 3.2.3, Eq. 8, page 494 | `CrossEntropyLoss` | cuvis-ai-unet |
| Combined loss `l_vl.ce` | Section 3.2.4, Eq. 10 | the trainer sums the three loss nodes | cuvis-ai-core |
| Voting, `vote(Yc)` | Section 3.3.1, Algorithm 1 and decoder step 1 | `VectorFieldVoteAccumulator` | cuvis-ai |
| Non-maximum suppression | Section 3.3.1, decoder step 2 | `LocalMaximaSuppression` | cuvis-ai |
| Vote threshold | Section 3.3.1, Eq. 12, decoder step 3 | `VotePeakCount`, `CentroidDetectionMetrics` | this plugin |
| Validation metrics | Section 5.2, Eq. 13 to 19, page 498 | `CentroidDetectionMetrics` | this plugin |

`ConnectedComponentInstances` has no counterpart in the paper. It turns a class mask into an
instance map for datasets whose annotations carry no object ids.

Not implemented in version 0.1 (decoder steps 4 to 7, page 495): collecting the border
coordinates of a centroid (Algorithm 2), fitting a shape through them, the class of an
instance with the probability threshold of 0.2, and the final instance and class maps. The
plugin therefore gives centres and counts, not instance masks. The border vectors are
trained and available on the `border_vectors` port for that later step.

## Differences from the paper

| Topic | Paper | This plugin | Reason |
|---|---|---|---|
| Class loss | Probabilities by dividing each logit by the sum of the logits (Eq. 3), then cross entropy (Eq. 8) | Softmax cross-entropy on the logits | The ratio of raw logits is undefined when they sum to zero and negative when signs differ. The reference implementation also calls `cross_entropy` on the logits. |
| IoU loss, MSE loss | Eq. 5, 9 and 11 | Not provided | The paper's best configuration on cell nuclei uses the vector loss with cross entropy, `VL-CE` (Section 6.2). |
| Objects cut by a crop | Not discussed. The reference implementation encodes the full image and crops the targets, so a cut object still points to its true centre | Cube and instance map are cropped together and encoded afterwards. Pixels whose nearest instance touches the crop edge get weight 0 in the vector loss | The crop node of Cuvis.AI carries a cube and one mask, not four target planes. Encoding a cut object would teach the centre of the visible fragment, so those pixels are left out. |
| Vote threshold | Chosen on the training set by exhaustive search (Section 6, page 499) | Chosen on the validation split, kept for the test split | Lentils have a test split besides validation. `CentroidDetectionMetrics` scores a list of thresholds in one pass. |
| Matching for precision and recall | Fitted instance against annotated instance, IoU above 0.1 (Section 5.2) | Vote peak against instance centroid, within `match_radius` pixels, one to one, strongest peak first | Without the shape fit there is no predicted instance to intersect. |
| Count | `TP + FP` (Eq. 19) | The same number, reported as `predicted_count`, plus `count_error`: the mean absolute difference between predicted and true count per frame | Over- and under-counting in different frames cancel in a total. |
| Minimum threshold | Not stated | 2 votes | A pixel whose vector is zero votes for itself, so every background pixel holds one vote. |
| Backbone | DeepLabV3+ with ResNet50, ResNet101 or Xception, and U-Net | `DynUNet` of cuvis-ai-unet | It exists as a node and takes any number of spectral bands. The paper reports an F1 score 0.7 % lower for its U-Net backbone on aerial crops (Section 6.1). |
| Input normalisation | `(x - 128) / 255` for 8-bit images (Section 5.1) | Per frame and band: subtract the mean, divide by the standard deviation (`Standardize`) | Hyperspectral cubes have no fixed value range. |

## Points the paper leaves open

| Point | Paper | Reference implementation at `ee01358` | This plugin |
|---|---|---|---|
| Number of output planes | 6 in Section 3.1, "five planes" in Section 3.2.1 | `4 + num_classes` | `4 + K`, `K = 2` in the shipped configs |
| Scale of the vectors | Not stated | Targets divided by `max_dist` for training | The backbone predicts vectors divided by `max_distance`; `CentroidNetHeads` multiplies them back to pixels, `VectorFieldLoss` divides the distance by `max_distance` |
| Clipping of the targets | "Roughly twice the average radius of the target objects" (Section 3.3.2); the rule is not given | Both vectors set to zero where the border vector is longer than `max_dist` | The same rule, parameter `max_distance`. It limits the border vector; a centre vector that is kept can be longer, by up to the distance from the border to the centroid |
| Which pixels are border pixels | Not stated | A 2 x 2 variance filter on the instance map | An instance pixel with a 4-neighbour of another id or outside the frame |
| Equal distances | Not stated | Left to `argmin` | Lower instance id for centroids, first border pixel in row-major order |
| Rounding of a vote to a pixel | Algorithm 1 adds the vector to the coordinate | `np.round` | Each vector component is rounded half to even (`torch.round`). A centre between pixels therefore spreads its votes over the neighbouring pixels |
| Votes that leave the image | Not stated | Dropped | Dropped |
| Suppression window | An `n x m` window, sizes not given | `min_distance = 7 // 2` | `window` of `LocalMaximaSuppression`; 15 in the lentils config, where the objects are larger |
| Equal maxima in one window | "Reduced to single points" | Left to `peak_local_max` | The first in row-major order stays (`keep_plateaus=False`) |
| Vote threshold default | Found by search | 30 | No default; chosen on validation |
| Optimiser | Adam, learning rate 0.001, early stopping (Section 5.1) | Adam, 0.001, batch 20 | Adam, 0.001 in the example |
| Training crop | 256 x 256 (Section 4.4) | 100 x 100 | 192 x 192 in the lentils config, four crops per frame |

## The lentils configuration

`centroidnet_fod_lentils_train.yaml` is set up for the public `Industrial_FOD_Lentils` set
(1000 x 1080 pixels, 61 bands, 808 training, 148 validation and 180 test frames in the
`adaclip` split). Its values come from the annotations of the training split:

- 500 of the 808 training frames contain objects, 1120 in total, at most 6 per frame.
- Half of the shorter bounding-box side, an upper bound of the inscribed radius, is 14.3
  pixels on average and at most 47 pixels. `max_distance` is 48, so no object loses the
  vectors of its interior. `match_radius` is 15.
- The set has class masks without object ids. `ConnectedComponentInstances` makes every
  connected region of one class one object, so two touching objects of the same class
  count as one.

## Cost of the encoder

`InstanceVectorFieldEncoder` runs on the device of its input, once per frame and instance.
On a 1000 x 1080 frame it took about 2 ms per instance on a laptop CPU (44 ms for 5
instances, 0.7 s for 300). For every instance it searches only the bounding box grown by
`max_distance`, and it splits its distance matrices into chunks of at most `max_pairwise`
elements. In training it sees crops, so the cost there is far lower.

## Not tested in version 0.1

- Full frames are processed in overlapping tiles by `DynUNet` outside training. The tiles
  are blended plane by plane, vector planes included. The chain tests cover a frame of
  240 x 260 pixels; there is no test that compares tiled and untiled vector fields. The
  vote threshold is chosen on validation frames that pass through the same tiles, so it
  is tuned to whatever the blending does to the votes.
- No trained weights ship with the plugin.
