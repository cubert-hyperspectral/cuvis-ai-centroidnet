# Changelog

## [Unreleased]

### Added

- Added `CentroidNetHeads`, which splits a backbone output into centre vectors, border vectors and class logits.
- Added `VotePeakCount`, the number of vote peaks per image at or above a vote threshold.
- Added `VectorFieldLoss`, the mean Euclidean vector distance in units of the vote range, with optional per-pixel weights.
- Added `ConnectedComponentInstances`, an instance map from a class mask.
- Added `InstanceVectorFieldEncoder`, the centre and border vector targets of an instance map, with loss weights that leave out objects cut by the frame edge.
- Added the `centroidnet_count` pipeline config.
