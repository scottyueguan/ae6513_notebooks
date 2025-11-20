# AE 6513 Jupyter Notebooks

This repository contains Jupyter notebooks for exercises and demonstrations used in **AE6513-Mathematical Principles of Planning and Decision-Making for Autonomy**.

Follow the steps below to configure your environment.

---
## 0. Prerequisites
You need to first install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your system,
for better package and environment management.

## 1. Open the Terminal

**macOS / Linux:**
- Terminal

**Windows (use one of the following):**
- PowerShell
- Command Prompt (`cmd.exe`) — works but less recommended

> ⚠ If using PowerShell or Command Prompt on Windows for the first time, you may need to initialize Conda:
> ```bash
> conda init powershell
> ```
> then close and reopen the terminal.

## 1. Create a Conda Environment

Create a new environment with Python 3.11:

```bash
conda create -n ae6513 python=3.11
```
Activate the environment:

```bash
conda activate ae6513
```

## 2. Install Required Packages

Install dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
Install Jupyter:
```bash
conda install jupyter
```

## 3. Launch Jupyter Notebook

Start the notebook server:
```bash
jupyter notebook
```
This should open a browser window where you can select and run course notebooks.
For Monday's demo, open `nash_demo.ipynb`.

## 4. Verify Installation
In the Jupyter notebook, you can start from the first cell and press `Shift + Enter` to run each cell sequentially to ensure everything is set up correctly.

At the end of the notebook, you should see an animation rendering the Nash policies of a pursuer trying to capture an evader.
