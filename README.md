# Can These Views Be One Scene? Evaluating Multiview 3D Consistency when 3D Foundation Models Hallucinate

Official code for the paper.

[**Paper (arXiv)**](https://arxiv.org/abs/2605.18754) &nbsp;&middot;&nbsp;
[**Project page**](https://mvp18.github.io/3d-consistency-metrics) &nbsp;&middot;&nbsp;
[**SysCON3D dataset**](https://huggingface.co/datasets/syscon3d-neurips26/syscon3d) &nbsp;&middot;&nbsp;
[**Human evaluation site**](https://mvp18.github.io/nvs-human-eval)

**Authors:** [Soumava Paul](https://mvp18.github.io)\*, [Prakhar Kaushik](https://toshi2k2.github.io)\*, [Alan Yuille](https://www.cs.jhu.edu/~ayuille1/) &nbsp;&mdash;&nbsp; CCVL, Johns Hopkins University. \*Equal contribution.

---

> ### Code release: **May 20, 5 PM EDT**
> v1 of this repository &mdash; the neural metric family
> (MEt3R, MASt3R-W-IMQ, RobustVGGT, &hellip;) and the COLMAP-based metrics
> &mdash; will be made public on **May 20, 5 PM EDT**.

---

## Hallucinated geometry on Gaussian noise

VGGT, MASt3R, DUSt3R, and Fast3R all return dense 3D structure when fed pure Gaussian noise as multi-view input. Metrics built on top (such as MEt3R) inherit this failure and report high 3D consistency on what is, in fact, no scene at all.

![Hallucinated reconstructions from Gaussian noise across four backbones](assets/gaussian_noise_hallucinations.gif)

In the [interactive gallery on the project page](https://mvp18.github.io/3d-consistency-metrics#interactive-syscon3d-demo), you can orbit these reconstructions and switch between scene types (Gaussian noise, cross-scene mixtures, one-outlier, patched corruptions, and clean baselines).

## Abstract

Multiview 3D evaluation assumes that the images being scored are observations of one static 3D scene. This assumption can fail in NVS and sparse-view reconstruction: inputs or generated outputs may contain artifacts, outlier frames, repeated views, or noise, yet still receive high 3D consistency scores. Existing reference-based metrics require ground truth, while ground-truth-free metrics such as MEt3R depend on learned reconstruction backbones whose failure modes are poorly characterized. We study this reliability problem by comparing neural reconstruction priors with classical geometric verification. We introduce SysCON3D, a controlled robustness benchmark for multiview 3D consistency, and a parametric family that decomposes neural metrics into backbone, residual, and aggregation components. This family recovers MEt3R and yields variants up to 3&times; more robust. Our analysis shows that VGGT, MASt3R, DUSt3R, and Fast3R can hallucinate dense geometry and cross-view support for unrelated scenes, repeated images, and random noise. We introduce COLMAP-based metrics that use matches, registration, dense support, and reconstruction failure as failure-aware consistency signals. On real NVS outputs and a structured human study, these metrics achieve up to 4&times; higher correlation with human judgments than MEt3R.

## Citation

```bibtex
@misc{paul2026viewssceneevaluatingmultiview,
  title  = {Can These Views Be One Scene? Evaluating Multiview 3D Consistency when 3D Foundation Models Hallucinate},
  author = {Soumava Paul and Prakhar Kaushik and Alan Yuille},
  year   = {2026},
  eprint = {2605.18754},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url    = {https://arxiv.org/abs/2605.18754}
}
```