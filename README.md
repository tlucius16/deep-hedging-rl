# Deep Hedging with Reinforcement Learning

This repository explores **reinforcement learning–based hedging strategies** for SPX & SPY options.  
We use Bloomberg/WRDS data (kept local), simulators, and RL agents to test risk management performance.  

---

## 📂 Project Structure

deep-hedging-rl/
│
├── data/ # (DO NOT push raw Bloomberg data, keep locally)
│ ├── raw/ # local-only (ignored in .gitignore)
│ └── processed/ # small sample/cleaned data for demos
│
├── notebooks/ # Jupyter/Colab notebooks
│ ├── 01_data_exploration.ipynb
│ ├── 02_simulator_dev.ipynb
│ ├── 03_rl_training.ipynb
│ └── 04_results_analysis.ipynb
│
├── src/ # main Python modules
│ ├── data_pipeline/ # wrds/bloomberg loaders, cleaning functions
│ ├── simulator/ # hedging environment (Gym API)
│ ├── rl_agent/ # RL models, training loop
│ ├── evaluation/ # metrics, risk attribution, plots
│ └── utils/ # helpers, config, logging
│
├── experiments/ # configs & logs for each run
│ ├── configs/ # YAML/JSON config files
│ └── logs/ # training logs, wandb/mlflow outputs
│
├── reports/ # paper drafts, figures, slides
│
├── requirements.txt # pinned dependencies
├── environment.yml # optional, for conda setup
├── .gitignore # makes sure raw data is excluded
├── README.md # project overview, setup, usage
└── LICENSE # license if publishing



---

## ⚙️ Setup

### Option 1: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate deep-hedging-rl


pip install -r requirements.txt
