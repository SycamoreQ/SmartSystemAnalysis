# GBDT-DBN Reliability Analysis for C-MAPSS

This project implements the GBDT-DBN reliability assessment framework from Zhang et al. (2025) on the NASA C-MAPSS (FD001) dataset.

## Setup

1.  **Data:**
    * Download the C-MAPSS dataset.
    * Create a folder `archive/FD001/`.
    * Place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` inside it.

2.  **Environment:**
    ```bash
    # Create a virtual environment (optional but recommended)
    use uv since I used uv so:
    uv init 
    uv venv
    source venv/bin/activate 
    
    # Install dependencies
    uv pip install -r requirements.txt
    ```

## How to Run the Pipeline

The pipeline is broken into numbered scripts inside the `src/` folder. **Run them in order from your project's root directory.**

1.  **Preprocess Data:**
    This script loads the raw `.txt` files, calculates RUL, creates 5 discrete health states (0=Healthy, 4=Failure), and saves the processed data.
    ```bash
    python src/preprocess.py
    ```

2.  **Train GBDT Monitor (Emission Model):**
    This trains the GBDT to predict the health state from sensors. It saves the calibrated model and the `P(Y|C)` (emission CPT) derived from its confusion matrix.
    ```bash
    python src/train_monitor.py
    ```

3.  **Build Weibull Model (Transition Model):**
    This fits a Weibull distribution to the training RULs to create an empirical aging model. It saves the `P(C'|C)` (transition CPT).
    ```bash
    python src/build_transition.py
    ```

4.  **Run DBN Inference:**
    This loads the test data, the GBDT, and both CPTs. It builds the DBN and runs inference on each test unit, saving the final reliability estimates.
    ```bash
    python src/run_inference.py
    ```

5.  **Plot Results (Optional):**
    This script loads the output from step 4 and plots the reliability curves for a few example units.
    ```bash
    python src/plot_results.py
    ```

6.  **Additional**
    I have also added preprocess_v2.py to implement rolling averages to improve sensor quality and making it less noisier . This improved accuracy drastically.