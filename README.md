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

## 2. Clone or Download the Course Repository
**Option A**: Clone via Git (recommended)
In the terminal, run:
```bash
git clone https://github.com/scottyueguan/ae6513_notebooks.git
```

Then move into the directory:
```bash
cd ae6513_notebooks
```

**Option B**: Download ZIP

1. Visit the repository:
https://github.com/scottyueguan/ae6513_notebooks

2. Click Code → Download ZIP

3. Extract the ZIP

4. Navigate to the extracted folder in terminal, e.g.:
    ```
    cd ~/put_in_your_path/ae6513_notebooks
    ```

## 3. Create a Conda Environment and Install Required Packages

Create a new environment with Python 3.11:

```bash
conda create -n ae6513 python=3.11
```
Activate the environment:

```bash
conda activate ae6513
```

Install dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
Install Jupyter:
```bash
conda install jupyter
```

Remember, you only need to create the environment and install the packages once. 
Next time when you want to run the Jupyter notebooks, just activate the environment with the command above and launch Jupyter Notebook.


## 5. Launch Jupyter Notebook

Start the notebook server:
```bash
jupyter notebook
```
This should open a browser window where you can select and run course notebooks.
For Monday's demo, open `game_example/nash_demo.ipynb`.

## 6. Verify Installation
In the Jupyter notebook, you can start from the first cell and press `Shift + Enter` to run each cell sequentially to ensure everything is set up correctly.

At the end of the notebook, you should see an animation rendering the Nash policies of a pursuer trying to capture an evader.
