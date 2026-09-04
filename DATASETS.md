# Datasets, licenses and known issues

CDIOD is not a new collection of images. It is a cross-domain incremental-learning *protocol*
built on three existing object-detection datasets — **DIOR**, **Pascal VOC 2012** and **RUOD** —
by partitioning their existing annotations into phases with disjoint category sets. We collected
no images and we created or edited no bounding box.

## 1. Provenance

| component | the version we used | original dataset | our changes |
|---|---|---|---|
| **DIOR** | the COCO JSON distributed by [CDFSOD-benchmark](https://github.com/lovelyqian/CDFSOD-benchmark): `train.json` (18,463 images / 152,602 boxes) and `test.json` (5,000 / 28,810), 20 classes. This is not DIOR's official trainval 11,725 / test 11,738 split, and the XML→COCO conversion is upstream of us — we never handled DIOR's original XML | Li, Wan, Cheng, Meng, Han, ISPRS JPRS 159:296–307, 2020 | `train.json` partitioned by category into phase files; `test.json` used unchanged as the validation split |
| **Pascal VOC** | the Roboflow export *"Pascal VOC 2012 – v1 raw"* (exported 2021-06-17), obtained as the `PascalVOC` subset of **ODinW-13** (GLIP / ELEVATER) following the preparation in [IVLOD / ZiRa](https://github.com/JarintotionDin/ZiRaGroundingDINO) — 13,690 train / 3,422 valid, 20 classes | Everingham, Van Gool, Williams, Winn, Zisserman, IJCV 88(2):303–338, 2010 (VOC **2012**) | removed the Roboflow dummy background category (`id: 0`, name `VOC`), which had no annotations attached — this is what `without_background` in the filenames refers to; partitioned by category |
| **RUOD** | the original COCO annotations from the official release, `RUOD_ANN/instances_train.json` and `instances_test.json` (official split 9,800 train / 4,200 test, 10 classes), per the [MMDetection](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md) / [LQIT](https://github.com/BIGWangYuDong/lqit/tree/main/configs/detection/ruod_dataset) preparation | Fu, Liu, Fan, Chen, Fu, Yuan, Zhu, Luo, Neurocomputing 517:243–256, 2023 | partitioned by category |

The phase files are exact partitions of the upstream annotations, with no annotation `id`
occurring in two phases of the same dataset and disjoint category sets per phase:
DIOR Task2-\* = Task4-\* = 152,602 = `train.json`; Pascal VOC Task2-\* = Task4-\* = 31,356 = the
Roboflow train file; RUOD Task2-1 + Task2-2 = 51,935 = `instances_train.json`.

Three of the files we redistribute are **byte-identical** to their upstream source and differ
only in name:

| our file | identical to |
|---|---|
| `DIOR/valid/annotations_without_background.json` | CDFSOD-benchmark `DIOR/annotations/test.json` |
| `RUOD/train/annotations_without_background.json` | RUOD `RUOD_ANN/instances_train.json` |
| `RUOD/valid/annotations_without_background.json` | RUOD `RUOD_ANN/instances_test.json` |

## 2. Licenses

Three layers with different terms. **The repository `LICENSE` does not cover the data.**

| layer | terms |
|---|---|
| Code in this repository | **Apache-2.0**, inherited from [MMDetection](https://github.com/open-mmlab/mmdetection) |
| The CDIOD phase annotation files and the scripts that produce them | **CC BY-NC 4.0**, matching the DIOR annotations they derive from |
| The images and the original annotations | each original licensor's terms, below. We hold no rights in them and cannot grant permission to use them |

- **DIOR — CC BY-NC 4.0, plus citation.** The authors' project page states the dataset is
  "freely available under the CC BY-NC 4.0 license agreement". Non-commercial research use is
  within those terms; commercial use is not. CDFSOD-benchmark, which redistributed the COCO
  conversion, is Apache-2.0 (code) and asks that the datasets be "cite[d] properly".
- **Pascal VOC — CC BY 4.0 as declared by the export; the original terms are narrower.** The
  Roboflow export declares CC BY 4.0, but that is the redistributor's declaration, not the VOC
  organizers'. The official VOC2012 "Database Rights" section states only that the data includes
  Flickr images whose terms of use must be respected, and that queries about use or ownership be
  addressed to the organizers. ODinW/ELEVATER adds: *"Please follow the original license of each
  dataset and keep this benchmark for academic purposes."*
- **RUOD — no written license accompanies the data.** `dlut-dimt/RUOD` carries an Apache-2.0
  `LICENSE`, but that repository holds only three files and no data; the mirror that hosts the
  archive (`xiaoDetection/RUOD`) has no license file; and neither the official READMEs nor the
  MMDetection / LQIT documentation states any usage restriction. This is an *absence* of a
  statement rather than a permissive one, and we cannot confirm terms on the authors' behalf. If
  you need a documented license for RUOD, please contact Fu et al. directly.

## 3. Download integrity

| file | bytes | SHA-256 |
|---|---|---|
| `CDIOD_annotations.tar.gz` | 11,118,532 | `9168e66c52f9849a451f10ffdaf6b1df41a9fa024fd3583299f58733b3b00c26` |
| `CDIOD_images.tar.gz` | 11,545,600,485 | `048e741613225d8356f42b9926b34946eaae0ffc595f0df0df5aa4ff3c8851ee` |

The first upload of `CDIOD_annotations.tar.gz` (9,834,340 bytes, SHA-256
`3d4f62cd2d1dbc0f5e255011920431be29d1faa2cf0a7e87aa7783027c8d33d2`) held 17 files and omitted
`RUOD/train/annotations_without_background.json`, which the 5-phase configuration loads; 5-phase
runs failed without it, 10-phase runs were unaffected. The current archive adds that file and is
otherwise identical. **If you downloaded the earlier archive, re-download.**

Annotation files inside the archive, under the prefix `data/CDIOD/`. `…` abbreviates
`annotations_without_background`.

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
| `RUOD/train/….json` | 16,997,328 | 9,800 | 51,935 | 10 | `ec124c587382c4e4c89d28490ec3a145497b27ac2ad2ffa0afbf263bdc406310` |
| `RUOD/valid/….json` | 7,487,412 | 4,200 | 22,969 | 10 | `b5dee6270231629b2c32fb21996fb325923257ba8caade8134b9d31a042b22fe` |

`RUOD/train/….json` is the unpartitioned RUOD training set used by the 5-phase setting, so its
51,935 annotations are the same ones as in RUOD Task2-1 + 2-2.

For the record, the DIOR COCO files we consumed from CDFSOD-benchmark, which publishes no
checksums of its own: `train.json` 24,401,269 bytes,
`888ee4588efb824c8ddcce87f9f04d6c47bf9ddae86f0fa00c6b9b088840335b`; `test.json` 4,697,409 bytes,
`6f446c7e24b8e4b92f9dc71127ca80ca0363a0037a428acd81de31aff500e28f`.

## 4. Known issues

Both are properties of the upstream datasets, inherited by CDIOD. We keep the released files
byte-identical to their sources rather than silently correcting them, because that fidelity is
what makes the splits reproducible.

**Byte-identical duplicate images.** DIOR and RUOD contain images identical to one another under
different file names and different `image_id`s. Among the images referenced by the released
annotation files:

| dataset | total groups | spanning train & valid | inside train only | inside valid only |
|---|---|---|---|---|
| DIOR | 41 | 12 | 28 | 1 |
| RUOD | 756 | 322 | 366 | 68 |
| Pascal VOC | 0 | 0 | 0 | 0 |
| **total** | **797** | **334** | **394** | **69** |

334 groups place a byte-identical image in both splits. This is not part of the CDIOD protocol:
the split membership is the upstream one, and 463 of the 797 groups lie entirely inside a single
split, which a category partition cannot create. We do not de-duplicate, since that would deviate
from the upstream splits our reported numbers are measured on. The group lists (keyed by SHA-256)
are available on request.

**Two annotations outside their declared image rectangle**, out of 479,465, both present unchanged
upstream:

| dataset | ann `id` | image | class | declared W×H | bbox (xywh) |
|---|---|---|---|---|---|
| DIOR | 5895 | `01072.jpg` | `harbor` | 800×800 | `[-1, 134, 184, 216]` |
| RUOD | 15514 | `008431.jpg` | `cuttlefish` | 1024×683 | `[0, 712, 587, 311]` |

The DIOR box is one pixel left of the image edge; MMDetection's `Resize` clips it
(`clip_object_border=True`). The RUOD case is an EXIF orientation inconsistency rather than an
out-of-image box: the stored JPEG is 683×1024 with `EXIF Orientation = 6`, so the declared
`width`/`height` is the post-rotation size while the box is in the as-stored frame, where it fits
(`587 ≤ 683`, `1023 ≤ 1024`). MMDetection validates against the declared size and drops this
annotation at load time.


