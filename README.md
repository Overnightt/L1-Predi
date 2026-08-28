# L1-Predi 

> **Work in progress**: this project is currently undergoing major changes (multi-league support, new features, architecture updates) and is **not usable in its current state**. Expect broken scripts and outdated instructions until this notice is removed.

A neural network-based predictor for football match outcomes, originally built for Ligue 1 and now being extended to the five major European leagues. This project uses historical match data to predict the probability of home wins, draws, and away wins.

## About the Model

### How It Works

L1-Predi uses a Long Short-Term Memory (LSTM) neural network architecture, which is better suited for sequential data like match history:

- **Input Layer**: Takes sequences of 26 past match statistics for both home and away teams
- **LSTM Layer**: 16 hidden units that process matches chronologically, retaining relevant information over time
- **Output Layer**: 3 neurons representing probabilities for each outcome (Home win, Draw, Away win)

The model was built using PyTorch and originally trained on **12249 matches** from the 2019-2026 seasons using data from the website [DataHub]([https://datahub.io/](https://datahub.io/football)).

## Version 2 Updates

The model has been extended with a hybrid architecture: the LSTM still processes the sequential match history, but its output is now combined with a set of static (non-sequential) features before the final prediction layer. The static features currently included are:

- **Head-to-head score**: a measure of historical dominance between the two teams facing each other, based on their past confrontations, weighted by how many such confrontations exist
- **Squad average age**: for both the home and away team, sourced from Transfermarkt and manually compiled into a dedicated dataset covering multiple seasons and leagues
- **Squad total market value**: for both the home and away team, from the same dataset

These static features are normalized (mean/standard deviation) using statistics computed strictly from the training data, to avoid leaking information from validation-period seasons into the training process.

The underlying match dataset has also been extended beyond Ligue 1 to cover all five major European leagues (Ligue 1, Serie A, Bundesliga, La Liga, Premier League), with the aim of eventually supporting cross-league predictions such as Champions League matches.

### A note on performance

Despite these additions, we have not observed a clear net improvement in prediction accuracy compared to the original single-league model. This is largely explained by the significant increase in training time introduced by the added data volume, with the multi-league expansion being the main contributing factor. Further work is needed to properly evaluate whether the new features provide a meaningful benefit once training is run for a sufficient number of epochs and averaged across multiple runs.

## Model Performance & Accuracy

### Key Metrics (original single-league model)

- **Validation Accuracy**: 55.6%
- **Average Prediction Accuracy**: 62%

### Understanding the Numbers

The model achieves 55.6% accuracy on validation data, which might seem modest at first glance. However, this is intentional. The model is designed to make **conservative predictions**, avoiding overconfident probability assignments to any single outcome.

**Important nuance**: When looking at the model's **most confident prediction** (the outcome with the highest probability), the accuracy is significantly higher. For example, during week 20 of the season, if you selected the outcome that the AI assigned the highest probability to, you would be correct **62% of the time**.

## How to Use the Model

### Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Pandas
- NumPy

### Running Predictions

1. **Navigate to the scripts directory**:
```bash
   cd ./scripts/
```

2. **Run the prediction script**:
```bash
   python ./predict.py
```

3. **Enter team names** when prompted:
   - You'll be asked to input the home team name
   - Then the away team name
   
   **Important**: Team names must be written **exactly** as they appear in the CSV data file. If you encounter an error, double-check your spelling and capitalization against the dataset.

### Example Usage

```
Home Team:Paris SG
Away Team:Marseille
Prediction for Paris SG vs Marseille
Home win: 50.35%
Draw    : 24.56%
Away win: 25.10%
```

---

## About This Project

This project was a learning experience where I developed skills in:

- **Artificial Intelligence**: Building and training neural networks for sports prediction
- **PyTorch**: Implementing deep learning models from scratch
- **Pandas**: Data manipulation, cleaning, and preprocessing of football statistics
- **Machine Learning Workflows**: From data collection to model deployment

The dataset required custom formatting and preprocessing to work with the neural network, providing hands-on experience with real-world data challenges.

---

## Disclaimer 

### Sports Betting Warning

**I am NOT responsible for any financial losses incurred from sports betting based on this model's predictions.** This project is for educational and entertainment purposes only. Sports betting involves significant financial risk, and this model should not be used as the sole basis for betting decisions.

**Gamble responsibly. Never bet more than you can afford to lose.**

## Use & Improve This Project

**Feel free to use this model as a basis for your own projects!** You can:
- Copy and modify the code for your own experiments
- Build upon this model with your own improvements
- Try different architectures or add new features
- Use it as a learning resource
- Simply take the Data for your own project

---
