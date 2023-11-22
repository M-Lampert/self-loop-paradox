# Self-loop Paradox

This repository contains the code for the extended abstract "The Self-loop Paradox: Investigating the Impact of Self-Loops on Graph Neural Networks" presented at the second Learning on Graphs conference (Virtual Event, November 27-30, 2023) by Moritz Lampert and Ingo Scholtes. The extended abstract is available on [OpenReview](https://openreview.net/forum?id=Urf6G7rk8A).

## Abstract

Many Graph Neural Networks (GNNs) add self-loops to a graph to include feature information about a node itself at each layer. However, if the GNN consists of more than one layer, this information can return to its origin via cycles in the graph topology. Intuition suggests that this “backflow” of information should be larger in graphs with self-loops compared to graphs without. In this work, we counter this intuition and show that for certain GNN architectures, the information a node gains from itself can be smaller in graphs with self-loops compared to the same graphs without. We adopt an analytical approach for the study of statistical graph ensembles with a given degree sequence and show that this phenomenon, which we call the self-loop paradox, can depend both on the number of GNN layers *k* and whether *k* is even or odd. We experimentally validate our theoretical findings in a synthetic node classification task and investigate its practical relevance in 23 real-world graphs.

## Setup

The code is written in Python `3.10.11`. To set up all the requirements in your environment, we recommend using **VSCode** and the **Dev Container** extension. The repo provides a fully configured Dev Container configuration that will automatically set up a Python environment with all the requirements. If you do not want to use the Dev Container, you can also set up the environment manually by installing the requirements in `requirements.txt` and adding the `src/` directory to your `PYTHONPATH`.

## Usage

You can rerun the experiments by executing the **Jupyter** notebooks in the `notebooks` directory. Each notebook corresponds to at least one visualization from the paper. The notebook `poster_example.ipynb` is an exception and does not contain any experiments that are part of the paper but instead explores the self-loop paradox in a simple example graph as was done on the poster that was presented at the conference.

If you prefer to run the experiments from the command line, you can use Jupyters `nbconvert` tool to convert the notebooks to Python scripts and then run them with `python` or execute them directly as shown below:
```bash
jupyter nbconvert --execute --inplace notebook_path.ipynb
```

The code produces the visualizations from the paper and saves the content from tables as CSV-files. Note that the code is not deterministic and the results may differ slightly from the ones in the paper. Additionally, the visualizations use the default settings from `matplotlib` and `seaborn` and will look different than the ones in the paper. 

## Results

The original results from the paper are available in the `results` directory. The directory contains the following files:
- `synthetic.csv`: Results for the synthetic node classification task. Corresponds to Figure 1 in the paper.
- `real.csv`: Results for the real-world node classification task. Corresponds to Figure 2 in the paper.
- `statistics_synthetic.csv`: Walk statistics for the synthetic graphs. Corresponds to Tables 1 and 2 in the paper.
- `statistics_real.csv`: Walk statistics for the real-world graphs. Corresponds to Table 3 in the paper.

Note that if you rerun the experiments, the results will be overwritten.
