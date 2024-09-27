from .base import BaseEvaluator
from datetime import datetime

import json

import numpy as np
from collections import defaultdict
from typing import List, Dict, Union, Optional

def validate_data(entry: Dict[str, Union[str, float, int]]) -> None:
    """Validate if data entry has required fields and valid types."""
    required_fields = {
        'ticker': str,
        'event_date': datetime,  # Assuming 'event_date' is a string format date
        'd+1_open': (int, float),
        'd+1_close': (int, float),
        'd+2_open': (int, float),
        'd+2_close': (int, float),
        'd+3_open': (int, float),
        'd+3_close': (int, float),
        'prediction': (int, float)
    }
    for field, expected_type in required_fields.items():
        if field not in entry or not isinstance(entry[field], expected_type):
            raise ValueError(f"Missing or invalid field: {field}. Expected type: {expected_type}")


def calculate_daily_returns(entry: Dict[str, Union[str, float, int]], position: str) -> List[float]:
    """
    Calculate daily returns for each day within the 3-day period.

    Args:
    entry (dict): The entry data for a given asset.
    position (str): 'long' or 'short'.

    Returns:
    list: A list of daily returns over the 3-day period.
    """
    try:
        if position == 'long':
            day1_return = (entry['d+1_close'] - entry['d+1_open']) / entry['d+1_open']
            day2_return = (entry['d+2_close'] - entry['d+2_open']) / entry['d+2_open']
            day3_return = (entry['d+3_close'] - entry['d+3_open']) / entry['d+3_open']
        elif position == 'short':
            day1_return = (entry['d+1_open'] - entry['d+1_close']) / entry['d+1_open']
            day2_return = (entry['d+2_open'] - entry['d+2_close']) / entry['d+2_open']
            day3_return = (entry['d+3_open'] - entry['d+3_close']) / entry['d+3_open']
        else:
            raise ValueError("Invalid position.")
        return [day1_return, day2_return, day3_return]
    except ZeroDivisionError:
        return [0.0, 0.0, 0.0]  # Handle division by zero in case of invalid data


class SharpeRatioCalculator:
    def __init__(self, method: str = 'uniform', risk_free_rate: float = 0.01, days_in_year: int = 252):
        """
        A class to calculate the Sharpe ratio.

        Args:
        method (str): Normalization method for weights ('uniform', 'l1', 'l2').
        risk_free_rate (float): The risk-free rate, default is 1% (0.01).
        days_in_year (int): Number of trading days in a year, default is 252.
        """
        if method not in ['uniform', 'l1', 'l2']:
            raise ValueError("Invalid method. Choose between 'uniform', 'l1', 'l2'.")
        self.method = method
        self.risk_free_rate = risk_free_rate
        self.days_in_year = days_in_year

    def calculate_weights(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize the scores to calculate weights based on the method chosen.

        Args:
        scores (np.ndarray): Scores for each asset.

        Returns:
        np.ndarray: Normalized weights.
        """
        if len(scores) == 0:
            raise ValueError("Score list is empty.")

        if self.method == 'l1':
            norm = np.sum(np.abs(scores))
        elif self.method == 'l2':
            norm = np.sqrt(np.sum(np.square(scores)))
        else:
            return np.ones_like(scores) / len(scores)

        return scores / norm if norm else np.ones_like(scores)

    def calculate_returns(self, data: List[Dict[str, Union[str, float, int]]]) -> Dict[str, np.ndarray]:
        """
        Calculate daily returns for long and short positions based on predictions.

        Args:
        data (list of dict): List of asset data.

        Returns:
        dict: Returns grouped by date.
        """
        returns_by_date = defaultdict(lambda: np.zeros(3))
        grouped_by_date = defaultdict(list)

        # Validate and group data by event date
        for entry in data:
            try:
                validate_data(entry)
                grouped_by_date[entry['event_date']].append(entry)
            except ValueError as e:
                print(f"Data validation error: {e}")
                continue

        # Filter dates with at least 10 entries
        filtered_data = {date: entries for date, entries in grouped_by_date.items() if len(entries) >= 10}

        for date, entries in filtered_data.items():
            # Sort entries by prediction
            sorted_entries = sorted(entries, key=lambda x: x['prediction'])

            # Take top 10% as long and bottom 10% as short
            long_positions = sorted_entries[-int(0.1 * len(sorted_entries)):]  # Top 10%
            short_positions = sorted_entries[:int(0.1 * len(sorted_entries))]   # Bottom 10%

            # Extract predictions to calculate weights
            long_scores = np.array([entry['prediction'] for entry in long_positions])
            short_scores = np.array([entry['prediction'] for entry in short_positions])

            # Calculate weights for long and short positions
            long_weights = self.calculate_weights(long_scores)
            short_weights = self.calculate_weights(short_scores)

            # Calculate and aggregate daily returns for long positions
            for long_entry, weight in zip(long_positions, long_weights):
                long_returns = calculate_daily_returns(long_entry, 'long')
                returns_by_date[date] += np.array(long_returns) * weight

            # Calculate and aggregate daily returns for short positions
            for short_entry, weight in zip(short_positions, short_weights):
                short_returns = calculate_daily_returns(short_entry, 'short')
                returns_by_date[date] += np.array(short_returns) * weight

        return returns_by_date

    def calculate_portfolio_sharpe(self, returns_by_date: Dict[str, np.ndarray]) -> float:
        """
        Calculate the portfolio-level Sharpe ratio based on weighted daily returns.

        Args:
        returns_by_date (dict): A dictionary with dates as keys and 3-day cumulative returns for each date.

        Returns:
        float: Sharpe ratio for the portfolio.
        """
        if not returns_by_date:
            raise ValueError("No returns data available for Sharpe ratio calculation.")

        # Flatten the list of daily returns into a single list
        all_daily_returns = [ret for returns in returns_by_date.values() for ret in returns]
        avg_return = np.mean(all_daily_returns)
        std_dev = np.std(all_daily_returns, ddof=1)

        if std_dev == 0:
            return 0  # Avoid division by zero

        # Annualized return and volatility based on 3-day returns
        annualized_return = (1 + avg_return) ** (self.days_in_year / 3) - 1
        annualized_volatility = std_dev * np.sqrt(self.days_in_year / 3)

        return (annualized_return - self.risk_free_rate) / annualized_volatility

    def __call__(self, data: List[Dict[str, Union[str, float, int]]]) -> Dict[str, float]:
        """
        Calculate the portfolio Sharpe ratio for different methods and risk-free rates.

        Args:
        data (list of dict): List of asset data.

        Returns:
        dict: Sharpe ratios for different combinations of methods and risk-free rates.
        """
        sharpe_ratios = {}

        # Define the risk-free rates and methods
        risk_free_rates = [0.04, 0.045, 0.05]
        methods = ['uniform', 'l1', 'l2']

        # Loop over each combination of method and risk-free rate
        for risk_free_rate in risk_free_rates:
            self.risk_free_rate = risk_free_rate

            for method in methods:
                self.method = method
                # Calculate returns grouped by date and asset
                returns_by_date = self.calculate_returns(data)

                # Calculate the Sharpe ratio for the portfolio
                sharpe_ratio = self.calculate_portfolio_sharpe(returns_by_date)
                sharpe_ratios[f"method_{method}_rf_{risk_free_rate}"] = sharpe_ratio

        return sharpe_ratios


class SharpeEvaluator(BaseEvaluator):
    score_data: Optional[List[Dict[str, Union[str, float, int]]]] = None
    save_dir: Optional[str] = None
    def _encode_dataset(self):
        if self.score_data is None:
            self.score_data = []


    def evaluate(self):
        evaluator = SharpeRatioCalculator()
        result = evaluator(self.score_data)

        json.dump(result, open('file_name.json', 'w'), ensure_ascii=False, indent=4)
