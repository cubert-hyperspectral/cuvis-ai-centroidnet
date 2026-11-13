# Changelog

## [Unreleased]

### Added

- Added `CentroidNetHeads`, which splits a backbone output into centre vectors, border vectors and class logits.
- Added `VotePeakCount`, the number of vote peaks per image at or above a vote threshold.
- Added `VectorFieldLoss`, the mean Euclidean vector distance in units of the vote range, with optional per-pixel weights.
- Added `ConnectedComponentInstances`, an instance map from a class mask.
- Added `InstanceVectorFieldEncoder`, the training targets of an instance map: centre and border vectors, an object mask for the class loss, and loss weights that leave out objects cut by the frame edge.
- Added `CentroidDetectionMetrics`: precision, recall, F1 and count error of vote peaks matched to instance centroids, for several vote thresholds at once, pooled over the epoch.
- Added the `centroidnet_fod_lentils_train` pipeline config (training, validation metrics and label-free counting in one graph) and `centroidnet_count`, its label-free part.
