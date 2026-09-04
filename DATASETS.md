# Datasets, licenses and known issues

CDIOD is not a new collection of images. It is a cross-domain incremental-learning *protocol*
built on top of three existing object-detection datasets — **DIOR**, **Pascal VOC 2012** and
**RUOD** — by partitioning each dataset's existing annotations into phases with disjoint
category sets. We did not collect images and we did not create or edit any bounding box.

This document records, for each component: where we obtained it, what we changed, the terms
under which its original licensor distributes it, how to verify your download, and the issues
we are aware of. If you are auditing this release, this file plus the checksums in §4 should
be enough to do so without contacting us.

---

## 1. Components and provenance

| component | the version we used | original dataset | our changes |
|---|---|---|---|
| **DIOR** | the COCO JSON distributed by [CDFSOD-benchmark](https://github.com/lovelyqian/CDFSOD-benchmark) (Fu et al., ECCV 2024) — `train.json` (18,463 images / 152,602 boxes) and `test.json` (5,000 / 28,810), 20 classes | Li, Wan, Cheng, Meng, Han, *Object detection in optical remote sensing images: A survey and a new benchmark*, ISPRS JPRS 159:296–307, 2020 | none to images or annotations. `train.json` partitioned by category into phase files; `test.json` used unchanged as the validation split |
| **Pascal VOC** | the Roboflow export *"Pascal VOC 2012 – v1 raw"* (exported 2021-06-17), obtained as the `PascalVOC` subset of **ODinW-13** (GLIP / ELEVATER), following the ODinW-13 preparation in [IVLOD / ZiRa](https://github.com/JarintotionDin/ZiRaGroundingDINO) — 13,690 train / 3,422 valid, 20 classes | Everingham, Van Gool, Williams, Winn, Zisserman, *The PASCAL Visual Object Classes (VOC) Challenge*, IJCV 88(2):303–338, 2010 (VOC **2012**) | removed the Roboflow dummy background category (`id: 0`, name `VOC`) — this is what `without_background` in the filenames refers to; no annotation was attached to it. Partitioned by category |
| **RUOD** | the original COCO annotations from the official release (`RUOD_ANN/instances_train.json`, `RUOD_ANN/instances_test.json`), per the [MMDetection](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md) / [LQIT](https://github.com/BIGWangYuDong/lqit/tree/main/configs/detection/ruod_dataset) preparation — official split 9,800 train / 4,200 test, 10 classes | Fu, Liu, Fan, Chen, Fu, Yuan, Zhu, Luo, *Rethinking general underwater object detection: Datasets, challenges, and solutions*, Neurocomputing 517:243–256, 2023 | none. Partitioned by category |

Two points worth stating explicitly:

- **The DIOR component was taken from CDFSOD-benchmark, already in COCO form.** We never handled
  DIOR's original XML annotations; the XML→COCO conversion happened entirely upstream of this
  work.
- **The DIOR split is not DIOR's official split.** CDFSOD-benchmark's train 18,463 / test 5,000
  (which together cover all 23,463 DIOR images) is inherited from Xiong & Liu, *CD-FSOD: A
  benchmark for cross-domain few-shot object detection*, ICASSP 2023 — not from DIOR's official
  trainval 11,725 / test 11,738.

The phase files are exact partitions of the upstream annotations, with no annotation `id`
occurring in two phases of the same dataset and disjoint category sets per phase:

| dataset | sum over phase files | equals upstream |
|---|---|---|
| DIOR | Task2-\* = Task4-\* = 152,602 | `train.json` = 152,602 |
| Pascal VOC | Task2-\* = Task4-\* = 31,356 | Roboflow train file = 31,356 |
| RUOD | Task2-1 + Task2-2 = 51,935 | `instances_train.json` = 51,935 |

For DIOR and RUOD, the files we redistribute are **byte-identical** to the upstream files and
differ only in name:

| our file | identical to |
|---|---|
| `DIOR/valid/annotations_without_background.json` | CDFSOD-benchmark `DIOR/annotations/test.json` |
| `RUOD/train/annotations_without_background.json` | RUOD `RUOD_ANN/instances_train.json` |
| `RUOD/valid/annotations_without_background.json` | RUOD `RUOD_ANN/instances_test.json` |

---

## 2. Licenses

**Three layers, with different terms. Please read this before assuming the repository
`LICENSE` covers the data.**

| layer | terms |
|---|---|
| Code in this repository | **Apache-2.0**, inherited from [MMDetection](https://github.com/open-mmlab/mmdetection) (see `LICENSE`) |
| The CDIOD phase annotation files and the scripts that produce them — i.e. the part that is ours | **CC BY-NC 4.0** |
| The images and the original annotations | governed by each original licensor's terms, below. We hold no rights in them |

We chose CC BY-NC 4.0 for our layer deliberately rather than for convenience: the DIOR
annotations our phase files are derived from are themselves CC BY-NC 4.0, so a more permissive
license on our layer would misstate what a downstream user is actually permitted to do.

**We cannot grant permission to use the images or the original annotations.** We do not hold
those rights. Each component's terms, as stated by its own licensor:

- **DIOR — CC BY-NC 4.0, plus citation.** The authors' project page states the dataset is
  "freely available under the CC BY-NC 4.0 license agreement" and asks that the ISPRS 2020
  paper be cited. Non-commercial research use is within those terms; commercial use is not.
  CDFSOD-benchmark, which redistributed the COCO conversion, is Apache-2.0 (code) and asks only
  that the datasets be "cite[d] properly".
- **Pascal VOC — CC BY 4.0 as declared by the export; the original terms are narrower.** The
  Roboflow export declares CC BY 4.0 in `README.dataset.txt` and in the COCO `licenses` block,
  but that is the redistributor's declaration, not the VOC organizers'. The official VOC2012
  "Database Rights" section states only: *"The VOC2012 data includes images obtained from the
  'flickr' website. Use of these images must respect the corresponding terms of use… Any queries
  about the use or ownership of the data should be addressed to the organizers."* The underlying
  image copyrights rest with the individual Flickr users. ODinW/ELEVATER adds: *"Please follow
  the original license of each dataset and keep this benchmark for academic purposes."*
- **RUOD — no written license accompanies the data.** The official repository `dlut-dimt/RUOD`
  contains an Apache-2.0 `LICENSE`, but that repository holds only three files and no data, and
  nothing states that the license covers the dataset. The mirror that actually hosts the archive
  (`xiaoDetection/RUOD`) has no license file. Neither the official READMEs nor the MMDetection /
  LQIT documentation states any usage restriction. This is an *absence* of a statement rather
  than a permissive one, and we cannot confirm terms on the authors' behalf. The only explicit
  requirement anywhere is citing the Neurocomputing 2023 paper. If you need a documented license
  for RUOD, please contact Fu et al. directly.

---

## 3. Citation

If you use CDIOD, please cite the three original datasets and the intermediate re-packagers
alongside our paper. Attribution is a license requirement for DIOR (CC BY-NC 4.0), not only a
courtesy.

```bibtex
@InProceedings{Wang_2026_CVPR,
  author    = {Wang, Xu and Lin, Zihan and Zhang, Yixin and Wang, Zilei},
  title     = {Boosting Vision-Language Models Towards Cross-Domain Incremental Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June}, year = {2026}, pages = {6249--6260}
}

@article{LI2020296,
  title   = {Object detection in optical remote sensing images: A survey and a new benchmark},
  author  = {Ke Li and Gang Wan and Gong Cheng and Liqiu Meng and Junwei Han},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {159}, pages = {296--307}, year = {2020},
  doi     = {10.1016/j.isprsjprs.2019.11.023}
}

@Article{Everingham10,
  author  = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
  title   = "The Pascal Visual Object Classes (VOC) Challenge",
  journal = "International Journal of Computer Vision",
  volume  = "88", number = "2", pages = "303--338", month = jun, year = "2010"
}

@article{fu2023rethinking,
  title     = {Rethinking general underwater object detection: Datasets, challenges, and solutions},
  author    = {Fu, Chenping and Liu, Risheng and Fan, Xin and Chen, Puyang and Fu, Hao
               and Yuan, Wanqi and Zhu, Ming and Luo, Zhongxuan},
  journal   = {Neurocomputing}, volume = {517}, pages = {243--256}, year = {2023},
  publisher = {Elsevier}
}

@inproceedings{fu2025cross,
  title     = {Cross-domain few-shot object detection via enhanced open-set object detector},
  author    = {Fu, Yuqian and Wang, Yu and Pan, Yixuan and Huai, Lian and Qiu, Xingyu
               and Shangguan, Zeyu and Liu, Tong and Fu, Yanwei and Van Gool, Luc and Jiang, Xingqun},
  booktitle = {European Conference on Computer Vision},
  pages     = {247--264}, year = {2025}, organization = {Springer}
}

@inproceedings{xiong2023cdfsod,
  title     = {CD-FSOD: A Benchmark for Cross-Domain Few-Shot Object Detection},
  author    = {Xiong, Wuti and Liu, Li},
  booktitle = {ICASSP}, year = {2023}
}

@inproceedings{li2021grounded,
  title     = {Grounded Language-Image Pre-training},
  author    = {Li, Liunian Harold and Zhang, Pengchuan and Zhang, Haotian and Yang, Jianwei
               and Li, Chunyuan and Zhong, Yiwu and Wang, Lijuan and Yuan, Lu and Zhang, Lei
               and Hwang, Jenq-Neng and Chang, Kai-Wei and Gao, Jianfeng},
  booktitle = {CVPR}, year = {2022}
}

@article{li2022elevater,
  title   = {ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models},
  author  = {Li, Chunyuan and Liu, Haotian and Li, Liunian Harold and Zhang, Pengchuan
             and Aneja, Jyoti and Yang, Jianwei and Jin, Ping and Lee, Yong Jae
             and Hu, Houdong and Liu, Zicheng and others},
  journal = {arXiv preprint arXiv:2204.08790}, year = {2022}
}
```

---

## 4. Download integrity

### Archives

| file | bytes | SHA-256 |
|---|---|---|
| `CDIOD_annotations.tar.gz` | 9,834,340 | `3d4f62cd2d1dbc0f5e255011920431be29d1faa2cf0a7e87aa7783027c8d33d2` |
| `CDIOD_images.tar.gz` | 11,545,600,485 | `048e741613225d8356f42b9926b34946eaae0ffc595f0df0df5aa4ff3c8851ee` |

### Supplementary file — required for the 5-phase setting

`CDIOD_annotations.tar.gz` contains 17 annotation files and **omits one**:
`RUOD/train/annotations_without_background.json`. The 5-phase (0-10) configuration trains its
RUOD phase on all ten RUOD classes at once and therefore loads this file, so 5-phase runs fail
without it. The 10-phase (0-5) configuration is unaffected. It is now published alongside the
archives at the same download links; place it in `data/CDIOD/RUOD/train/`.

| file | bytes | images | annotations | categories | SHA-256 |
|---|---|---|---|---|---|
| `RUOD/train/annotations_without_background.json` | 16,997,328 | 9,800 | 51,935 | 10 | `ec124c587382c4e4c89d28490ec3a145497b27ac2ad2ffa0afbf263bdc406310` |

### Annotation files inside `CDIOD_annotations.tar.gz` (prefix `data/CDIOD/`)

| path | bytes | images | annotations | categories | SHA-256 |
|---|---|---|---|---|---|
| `DIOR/train/…_Task2-1.json` | 4,775,235 | 9,689 | 26,954 | 10 | `0c1248d09c05d216e89366919e8c3d5cbddb4fc210a0d0fc76633acd48475498` |
| `DIOR/train/…_Task2-2.json` | 19,922,163 | 13,025 | 125,648 | 10 | `3a72ca4167ef89ed839d2b0ff0ef6d23822176f5d9ab29fdfb4109fb5b622288` |
| `DIOR/train/…_Task4-1.json` | 3,122,809 | 5,750 | 17,928 | 5 | `a5a2aac2edab8c94db301e84f4937ab548e3c02991dd5b91aaeb76b5f68a202c` |
| `DIOR/train/…_Task4-2.json` | 1,707,375 | 4,726 | 9,026 | 5 | `dbb599014ad3a762c87da83ca15aa266bdf8e17f4e9079e0f6836a0d2003fd48` |
| `DIOR/train/…_Task4-3.json` | 8,577,478 | 5,849 | 53,904 | 5 | `8b1103f3465d1b8f67db68cf5316d2af76cb8597b8ef10ca28a34fedceed0501` |
| `DIOR/train/…_Task4-4.json` | 11,483,044 | 9,161 | 71,744 | 5 | `571feb7ed30daf0444b00f14e51d42da676ab6167d7ee15bddb0b906b663e26c` |
| `DIOR/valid/….json` | 4,697,409 | 5,000 | 28,810 | 20 | `6f446c7e24b8e4b92f9dc71127ca80ca0363a0037a428acd81de31aff500e28f` |
| `PascalVOC/train/…_Task2-1.json` | 2,465,200 | 5,847 | 11,303 | 10 | `891c2f7b89604bc95fff5e9d17a5622c98ec11793b54733a9ae767d89e1b96ac` |
| `PascalVOC/train/…_Task2-2.json` | 4,319,882 | 9,928 | 20,053 | 20 | `d11fedc97ba00a1ce88558d1f58fea46527b9aa237cbb1eaa37a7a24c1eae0fa` |
| `PascalVOC/train/…_Task4-1.json` | 1,074,128 | 2,759 | 4,639 | 5 | `bf557c3414d23f212c0588ad61a0ecb332aa33426c3e882723c99102c138602a` |
| `PascalVOC/train/…_Task4-2.json` | 1,452,785 | 3,440 | 6,664 | 5 | `8a1d5638295ae831e72577425e2b920e6b3864ee9d8e6b493680eaae573203f4` |
| `PascalVOC/train/…_Task4-3.json` | 3,613,665 | 8,672 | 16,271 | 5 | `1c7281df2df58828c43af6bf49002c76e344c1df8c7d87b85da5a3445071b604` |
| `PascalVOC/train/…_Task4-4.json` | 861,867 | 2,147 | 3,782 | 5 | `fbccc3fb840f6fd079b552c87269a83d5e1a594c311ac10ab5cdbca158f04411` |
| `PascalVOC/valid/….json` | 1,593,682 | 3,422 | 7,835 | 20 | `31254ead511b0a6beb5199e5b78990991ca3beca6bf0ccbfc72328c95373cc0b` |
| `RUOD/train/…_Task2-1.json` | 4,358,222 | 5,189 | 32,984 | 5 | `86653fb691e484e3ff425bb81d342117d814dbaaed4efb1eda23b04aebc003c4` |
| `RUOD/train/…_Task2-2.json` | 2,768,260 | 6,376 | 18,951 | 5 | `38905b95c7ceb8127a15afe76cf8124d180a4528db09944c5f98b48873f9cc18` |
| `RUOD/valid/….json` | 7,487,412 | 4,200 | 22,969 | 10 | `b5dee6270231629b2c32fb21996fb325923257ba8caade8134b9d31a042b22fe` |

479,465 annotations across the 17 files. `…` abbreviates
`annotations_without_background`.

### Source files we consumed

CDFSOD-benchmark publishes no checksums, so these are ours. If your copy hashes to these
values, the DIOR component you have is the same one this release was built from.

| source file | bytes | SHA-256 |
|---|---|---|
| `DIOR/annotations/train.json` | 24,401,269 | `888ee4588efb824c8ddcce87f9f04d6c47bf9ddae86f0fa00c6b9b088840335b` |
| `DIOR/annotations/test.json` | 4,697,409 | `6f446c7e24b8e4b92f9dc71127ca80ca0363a0037a428acd81de31aff500e28f` |

---

## 5. Known issues

All of the following are properties of the upstream datasets, inherited by CDIOD. We keep the
released files byte-identical to their sources rather than silently correcting them, because
that fidelity is what makes the splits reproducible. They are documented here so that users can
account for them.

### 5.1 Byte-identical duplicate images

DIOR and RUOD both contain images that are byte-identical to one another while carrying
different file names and different `image_id`s. Counting duplicate groups among the images
referenced by the released annotation files:

| dataset | total groups | spanning train & valid | inside train only | inside valid only |
|---|---|---|---|---|
| DIOR | 41 | 12 | 28 | 1 |
| RUOD | 756 | 322 | 366 | 68 |
| Pascal VOC | 0 | 0 | 0 | 0 |
| **total** | **797** | **334** | **394** | **69** |

797 groups comprise 1,725 files, i.e. 928 redundant copies.

**334 groups place a byte-identical image in both the training and the validation split.** This
is not intentional and is not part of the CDIOD protocol. It is inherited: most of the
duplication does not cross the split boundary at all (which a partition cannot create), and the
split membership is upstream — our RUOD annotation files are byte-identical to the official
release, and our DIOR train/valid image sets are exactly those of CDFSOD-benchmark's
`train.json` / `test.json`, whose file-name intersection is 0.

We do not de-duplicate the release, because that would deviate from RUOD's official split and
from the CD-FSOD split, and the numbers in our paper are measured on the splits as released. If
you need to account for the overlap, the duplicate group lists (keyed by SHA-256) are available
on request.

### 5.2 Two annotations outside their declared image rectangle

Two annotations out of 479,465. Both are present unchanged in the upstream sources.

**DIOR — annotation `id` 5895, `01072.jpg`, class `harbor`, `bbox = [-1, 134, 184, 216]` on an
800×800 image.** The left edge is one pixel outside the image. This annotation is present with
identical `id`, `image_id` and `bbox` in CDFSOD-benchmark's `train.json`, where it is the only
out-of-bounds annotation. It appears in two released files (`Task2-2` for the 5-phase split and
`Task4-3` for the 10-phase split) because it is one annotation reachable from both splits.
MMDetection retains it (`inter_w = 183`, `inter_h = 216`) and `Resize`, whose
`clip_object_border` defaults to `True`, then clips `x` from −1 to 0.

**RUOD — annotation `id` 15514, `008431.jpg`, class `cuttlefish`, `bbox = [0, 712, 587, 311]`,
declared `width: 1024, height: 683`.** This one is an EXIF orientation inconsistency, not a box
outside the image. The stored JPEG is **683×1024** and carries `EXIF Orientation = 6`; applying
the orientation yields 1024×683. The COCO record declares the *post-rotation* size, while the
box is expressed in the *as-stored* frame, where it fits exactly: `0 + 587 ≤ 683` and
`712 + 311 = 1023 ≤ 1024`. The `width`/`height` fields and the box coordinates therefore refer
to two different orientations of the same raster. The correct handling is **not** to apply the
EXIF orientation: decode the raw JPEG (as OpenCV, and hence MMCV, does by default) and take the
image size from the decoded array rather than from the declared fields. Because MMDetection
validates against the declared size, it computes `inter_h = max(0, min(1023, 683) − 712) = 0`
and discards this annotation at load time — 1 of 51,935 RUOD training boxes; it never reaches
the model. The RUOD validation file contains no out-of-bounds annotations, so evaluation is
unaffected.

### 5.3 The 5-phase setting needs a file that the initial archive omitted

See §4. `RUOD/train/annotations_without_background.json` was missing from the first upload of
`CDIOD_annotations.tar.gz` and is now published alongside it. If you downloaded the archive
before this note was added, fetch that one file; nothing else changed and the archive checksums
above are unchanged.



