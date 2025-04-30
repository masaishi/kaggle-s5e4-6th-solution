# PS-S5E4: Podcast Listening Time Prediction

## Overview
This repository contains a machine learning solution for the Playground Series Season 5 Episode 4 competition, focused on predicting podcast listening time based on various podcast features.

## Features
- Prediction of listener duration for podcasts
- Feature engineering for podcast metadata and content attributes
- Multiple model implementations (LightGBM, XGBoost, TabNet, HGBR, SVR)
- Weights & Biases integration for experiment tracking
- Cross-validation with 7 folds

## Data Processing
- Efficient DataFrame operations using Polars
- Handling of categorical variables with advanced encoding techniques
- Feature engineering for time-based features, interaction features, and more

## Models
- Primary model: LightGBM
- Alternative implementations:
  - XGBoost
  - TabNet
  - Histogram-based Gradient Boosting Regressor
  - Support Vector Regressor

## Project Structure
- `notebooks/`: EDA and experimentation notebooks
- `src/`: Source code
  - `data/`: Data processing and feature engineering
  - `models/`: Model implementations
  - `config.py`: Configuration parameters
  - `main.py`: Main execution script
  - `utils.py`: Utility functions

## Recent Performance
Recent validation scores around 11.9 RMSE.

## Usage
1. Configure parameters in `src/config.py`
2. Run training using `src/main.py`
3. Track experiments in Weights & Biases