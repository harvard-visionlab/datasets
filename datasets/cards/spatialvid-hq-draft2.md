# SpatialVID-HQ (lab build)

## oroginal source dataset

[SpatialVID-HQ](https://huggingface.co/datasets/FelixYuan/SpatialVID-HQ) is a video dataset curated by Wang et al. (2026) ([project](https://huggingface.co/SpatialVID)).

The dataset contains XX video clips from YY videos (X hours, Y frames), and for each clip the authors provide several annotations, including an estimate of camera position and rotation at keyframes (5Hz).

```bibtex
@inproceedings{wang2026spatialvid,
  title={Spatialvid: A large-scale video dataset with spatial annotations},
  author={Wang, Jiahao and Yuan, Yufeng and Zheng, Rujie and Lin, Youtian and Gao, Jian and Chen, Lin-Zhuo and Bao, Yajie and Zeng, Chang and Zhou, Yanxi and Long, Xiao-Xiao and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={42592--42603},
  year={2026}
}
```

## visionlab build

Of course our copy is written to [slipstream](https://github.com/harvard-visionlab/slipstream) format for optimal dataloading.

The source dataset does not have a train/val/test split, so we created our own stratifying acros <XYZ>, with the restriction that all clips from a single youtubeID would be included in the same split (see Methods below).

<insert table with counts (number of channels, number of videos, number of clips)>

In addition, the full dataset contains clips where the movement appears to be carried by different sources, e.g., person walking, riding a bike, driving a car, a train, a boat, a horse, or from a drone. So we created a subset that's focused on the person-walking point of view (`subset=<name-of-subset>`, see methods) including a train/val/test split of its own.

<insert table with counts of splits>

In addition to providing the raw video frames (at different resolutions and frame rates), our samples include the source camera position information (<variable names>) linearly interpolated for video frames that fall between keyframes, which can be converted to egomotion values (<variable names>).

```python
# <show loading a dataset, and a sample>
```

The sample has the following fields:

- <field-name>: brief description
- <field-name>: brief description

## Usage

<here you should explain the visionlab.datasets api for accessing subsests of the dataset, use examples to illustrate the options available>

## Methods

<please write the methods as if the user was submitting a paper to NeurIPS or similar conference. Write two versions, one for the full dataset, and one for our person subset.>
