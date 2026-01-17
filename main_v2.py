import argparse
import sys
from pathlib import Path

from models.ml_model import CryptoEntryModel
from models.hyperparameter_optimizer import HyperparameterOptimizer, OptimizationMetric


def optimize_hyperparameters(symbol: str, timeframe: str, model_type: str, 
                            optimization_level: str, n_trials: int = 50):
    """
    Optimize hyperparameters for the specified model configuration.
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        timeframe: Candle timeframe (e.g., 15m)
        model_type: Model type (xgboost, lightgbm, random_forest)
        optimization_level: Optimization profile (conservative, balanced, aggressive)
        n_trials: Number of optimization trials
    """
    print("="*60)
    print(f"Optimizing hyperparameters for {symbol} ({timeframe})")
    print(f"Model: {model_type}")
    print(f"Optimization Level: {optimization_level}")
    print(f"Trials: {n_trials}")
    print("="*60)
    print()

    try:
        import optuna
    except ImportError:
        print("Error: Optuna not installed.")
        print("Install with: pip install optuna")
        sys.exit(1)

    model = CryptoEntryModel(
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
        optimization_level=optimization_level,
        use_multi_timeframe=True,
        use_feature_selection=True
    )

    print("Step 1: Loading and engineering features...")
    model.load_data()
    model.engineer_features()
    X, y, feature_names = model.prepare_training_data()

    print(f"Training data shape: {X.shape}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Positive samples: {(y == 1).sum()}")
    print(f"Negative samples: {(y == 0).sum()}")
    print()

    print("Step 2: Running hyperparameter optimization...")
    print()

    metric = OptimizationMetric.BALANCED
    optimizer = HyperparameterOptimizer(model_type=model_type, metric=metric)
    result = optimizer.optimize(X, y, n_trials=n_trials, verbose=True)

    optimizer.print_results_summary(result)

    results_path = Path(f"models/cache/optimization/{symbol}_{timeframe}_{model_type}_results.json")
    optimizer.save_results(str(results_path))

    print(f"\nOptimization results saved to: {results_path}")
    print(f"\nTo use these parameters, update the hyperparams in ml_model.py:")
    print(f"  'max_depth': {result.best_params.get('max_depth')},")
    print(f"  'learning_rate': {result.best_params.get('learning_rate'):.4f},")
    print(f"  'n_estimators': {result.best_params.get('n_estimators')},")

    return result


def train_with_optimized_params(symbol: str, timeframe: str, model_type: str,
                               optimization_level: str, results_path: str = None):
    """
    Train model using previously optimized hyperparameters.
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        timeframe: Candle timeframe (e.g., 15m)
        model_type: Model type (xgboost, lightgbm, random_forest)
        optimization_level: Optimization profile
        results_path: Path to optimization results JSON file
    """
    print("="*60)
    print(f"Training model with optimized parameters")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Model: {model_type}")
    print("="*60)
    print()

    model = CryptoEntryModel(
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
        optimization_level=optimization_level,
        use_multi_timeframe=True,
        use_feature_selection=True
    )

    if results_path:
        optimizer = HyperparameterOptimizer(model_type=model_type)
        results = optimizer.load_results(results_path)
        optimized_params = results.get('best_params', {})
        if optimized_params:
            model.hyperparams.update(optimized_params)
            print(f"Loaded optimized parameters from {results_path}")
            print(f"Parameters: {model.hyperparams}")
            print()

    print("Step 1: Loading data...")
    model.load_data()

    print("Step 2: Engineering features...")
    model.engineer_features()

    print("Step 3: Training model...")
    training_results = model.train()

    print()
    print("="*60)
    print("Training Results:")
    print("="*60)
    for key, value in training_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("="*60)

    return training_results


def compare_optimization_levels(symbol: str, timeframe: str, model_type: str):
    """
    Compare performance across different optimization levels.
    
    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        model_type: Model type
    """
    print("="*60)
    print(f"Comparing optimization levels for {symbol} ({timeframe})")
    print("="*60)
    print()

    levels = ['conservative', 'balanced', 'aggressive']
    results_comparison = {}

    for level in levels:
        print(f"\nTraining with {level} optimization...")
        print("-"*60)
        
        model = CryptoEntryModel(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            optimization_level=level,
            use_multi_timeframe=True,
            use_feature_selection=True
        )

        model.load_data()
        model.engineer_features()
        results = model.train()
        results_comparison[level] = results

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    metrics = ['test_precision', 'test_recall', 'test_accuracy']
    
    for metric in metrics:
        print(f"\n{metric}:")
        for level in levels:
            value = results_comparison[level].get(metric, 0)
            print(f"  {level}: {value:.4f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for crypto trading model")
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--timeframe', default='15m', help='Candle timeframe')
    parser.add_argument('--model-type', default='xgboost', 
                       choices=['xgboost', 'lightgbm', 'random_forest'],
                       help='Model type')
    parser.add_argument('--opt', default='balanced',
                       choices=['conservative', 'balanced', 'aggressive'],
                       help='Optimization level')
    parser.add_argument('--mode', default='optimize',
                       choices=['optimize', 'train', 'compare'],
                       help='Execution mode')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--results-path', help='Path to optimization results JSON')

    args = parser.parse_args()

    if args.mode == 'optimize':
        optimize_hyperparameters(
            symbol=args.symbol,
            timeframe=args.timeframe,
            model_type=args.model_type,
            optimization_level=args.opt,
            n_trials=args.n_trials
        )
    elif args.mode == 'train':
        train_with_optimized_params(
            symbol=args.symbol,
            timeframe=args.timeframe,
            model_type=args.model_type,
            optimization_level=args.opt,
            results_path=args.results_path
        )
    elif args.mode == 'compare':
        compare_optimization_levels(
            symbol=args.symbol,
            timeframe=args.timeframe,
            model_type=args.model_type
        )
