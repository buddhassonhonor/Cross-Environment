# Entropy-Smoothed Robust Reweighting with Conditional Feature Alignment

This repository contains the public research artifact for the manuscript
*Entropy-Smoothed Robust Reweighting with Conditional Feature Alignment for
Domain Generalization*.

## Contents

- `exp/`: experiment runners, model code, result summaries, and diagnostic plots.
- `scripts/`: helper scripts for generating manuscript tables and figures.
- `figures/`: final figures used by the manuscript.

## Code Availability

The experiment code, model implementations, result-processing scripts, and
plotting utilities supporting the study are available in this repository.

## Data Availability

The repository includes processed result files, generated figures, and metadata
needed to support the reported analyses. The experiments use publicly available
benchmark datasets: MNIST-derived benchmarks can be obtained from the standard
MNIST distribution, Breast Cancer Wisconsin is available from public
machine-learning repositories, and PACS is available from its public dataset
release. Large raw benchmark files and locally generated temporary caches are
not required for interpreting the reported results.

## Reproducibility Notes

The main controlled-benchmark entry points are under `exp/`, including the
experiment runners, PACS linear-probing scripts, sensitivity analysis, runtime
measurement, and plotting scripts. Result summaries used by the manuscript are
stored under `exp/results/` and `exp/linear_probe/`.
