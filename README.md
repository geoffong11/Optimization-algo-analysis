# Project Overview

This project aims to analyze modern first-order optimization algorithms: AdaGrad, RMSProp and Adam. These algorithms are foundational in the field of optimization for machine learning and have influenced the development of many subsequent techniques. By understanding these algorithms, we can better appreciate their strengths, weaknesses, and the contexts in which they are most effective. This analysis not only deepens comprehension of their mechanics but also highlights how they lay the groundwork for other advanced optimization methods.

For detailed information about the project, please refer to the ProjectReport.

# Project Report
The Project Report contains:
- An in-depth explanation of the algorithms analyzed
- Experimental setup and results
- Insights and conclusions drawn from the study

[Read the Project Report](ProjectReport.pdf)

# Code Folder
The Code Folder provides the implementation and experimentation scripts used in analysis. It includes:
1. **Setup**: Scripts to set up the environment and dependencies
    - Script names: `requirements.txt`
2. **Algorithms**: Python implementations of mini-batch, Momentum, AdaGrad, RMSProp, Adam
    - Script names: `GDAdam.py`, `GDAdaGrad.py`, `GDMiniBatch.py`, `GDMomentum.py`, `GDRMSProp.py`
3. **Experiments**: Scripts to run comparisons and generate results
    - Script names: `NeuralNetwork.py`, `NeuralTraining.py`, `Weightlayer.py`, `functions.py`
4. **Results**: Code to visualize and analyze optimizer performance
    - Script names: `graph.py`, `CustomPlot.py`, `MSEPlot.py`

## How to use the code
1. **Install Dependencies:**

    Navigate to the `code` directory and ensure all required packages are installed. Use the provided `requirements.txt` to set up the environment:
    ```
    cd code
    pip install -r requirements.txt
    ```
2. **Run Experiments:**
    Execute the `main.py` script:
    ```
    python main.py
    ```
