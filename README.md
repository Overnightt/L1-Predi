# L1-Predi 

A neural network-based predictor for Ligue 1 football match outcomes. This project uses historical match data to predict the probability of home wins, draws, and away wins.

---

## About the Model

### How It Works

L1-Predi uses a Long Short-Term Memory (LSTM) neural network architecture, which is better suited for sequential data like match history:

- **Input Layer**: Takes sequences of 26 past match statistics for both home and away teams
- **LSTM Layer**: 16 hidden units that process matches chronologically, retaining relevant information over time
- **Output Layer**: 3 neurons representing probabilities for each outcome (Home win, Draw, Away win)

The model was built using PyTorch and trained on **2,960 matches** from the 2017-2026 seasons using data from [DataHub's French Ligue 1 dataset](https://datahub.io/core/french-ligue-1).


---

## 📈 Model Performance & Accuracy

### Key Metrics

- **Validation Accuracy**: 55.6%
- **Average Prediction Accuracy**: 62%

### Understanding the Numbers

The model achieves 52.1% accuracy on validation data, which might seem modest at first glance. However, this is intentional. The model is designed to make **conservative predictions**, avoiding overconfident probability assignments to any single outcome.

**Important nuance**: When looking at the model's **most confident prediction** (the outcome with the highest probability), the accuracy is significantly higher. For example, during week 20 of the season, if you selected the outcome that the AI assigned the highest probability to, you would be correct **62% of the time**.

---

##  How to Use the Model

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
   
   ⚠️ **Important**: Team names must be written **exactly** as they appear in the CSV data file. If you encounter an error, double-check your spelling and capitalization against the dataset.

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

---
