"""Blocker hyperparameter optimizer using Optuna."""

import logging
from collections.abc import Callable
from typing import Any

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

logger = logging.getLogger(__name__)


class BlockerOptimizer:
    """Optimize VectorBlocker hyperparameters using Optuna.

    This optimizer tunes blocker hyperparameters such as embedding model
    and k_neighbors to maximize a given metric (e.g., BCubed F1 score).

    It supports:
    - Categorical parameters (e.g., embedding model names)
    - Integer parameters (e.g., k_neighbors range)
    - wandb integration for experiment tracking
    - Maximize or minimize optimization

    Example:
        # Define objective function
        def objective(params: dict) -> float:
            blocker = VectorBlocker(
                embedding_model=params["embedding_model"],
                k_neighbors=params["k_neighbors"],
                ...
            )
            # Run pipeline and return metric
            f1_score = evaluate_pipeline(blocker, train_data, gold_labels)
            return f1_score

        # Define search space
        search_space = {
            "embedding_model": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            "k_neighbors": (5, 50),
        }

        # Optimize
        optimizer = BlockerOptimizer(
            objective_fn=objective,
            search_space=search_space,
            direction="maximize",
            n_trials=20,
            wandb_kwargs={"project": "langres", "entity": "myteam"}
        )

        best_params = optimizer.optimize()
        print(f"Best parameters: {best_params}")
        # Output: {"embedding_model": "all-mpnet-base-v2", "k_neighbors": 35}

    Note:
        The objective function should take a dict of hyperparameters and
        return a single float metric value. Higher values are better for
        direction="maximize", lower for direction="minimize".

    Note:
        Search space supports:
        - Categorical: list of values (e.g., ["model-a", "model-b"])
        - Integer: tuple of (min, max) (e.g., (5, 50))
    """

    def __init__(
        self,
        objective_fn: Callable[[dict[str, Any]], float | dict[str, float]],
        search_space: dict[str, Any],
        primary_metric: str = "value",
        direction: str = "maximize",
        n_trials: int = 50,
        wandb_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize BlockerOptimizer.

        Args:
            objective_fn: Function that takes hyperparameters dict and
                returns either:
                - float: Single metric value (legacy, backwards compatible)
                - dict[str, float]: Dict of all metrics, where primary_metric
                  key specifies which to optimize. Other metrics are logged
                  as Optuna user attributes for wandb tracking.
            search_space: Dict defining parameter ranges:
                - For categorical: {"param_name": ["value1", "value2", ...]}
                - For integer: {"param_name": (min, max)}
            primary_metric: Key in the dict returned by objective_fn that
                specifies which metric to optimize. Only used when objective_fn
                returns dict. Default: "value". This enables clean separation:
                objective computes all metrics, config specifies which to optimize.
            direction: Optimization direction - "maximize" or "minimize".
                Use "maximize" for metrics like F1 score, "minimize" for
                metrics like error rate. Default: "maximize".
            n_trials: Number of optimization trials to run. Default: 50.
            wandb_kwargs: Optional dict of wandb.init() arguments for
                experiment tracking. If None, wandb integration is disabled.
                Example: {"project": "langres", "entity": "myteam"}

        Raises:
            ValueError: If direction is not "maximize" or "minimize".
            ValueError: If search_space is empty.

        Example:
            optimizer = BlockerOptimizer(
                objective_fn=my_objective,
                search_space={
                    "embedding_model": ["model-a", "model-b"],
                    "k_neighbors": (5, 50)
                },
                direction="maximize",
                n_trials=20
            )
        """
        if direction not in ["maximize", "minimize"]:
            raise ValueError(f"direction must be 'maximize' or 'minimize', got: {direction}")

        if not search_space:
            raise ValueError("search_space cannot be empty")

        self.objective_fn = objective_fn
        self.search_space = search_space
        self.primary_metric = primary_metric
        self.direction = direction
        self.n_trials = n_trials
        self.wandb_kwargs = wandb_kwargs

    def optimize(self) -> dict[str, Any]:
        """Run optimization and return best hyperparameters.

        This method runs Optuna optimization for n_trials, using the
        search space and objective function provided during initialization.
        If wandb_kwargs was provided, trials are logged to wandb.

        Returns:
            Dictionary of best hyperparameters found during optimization.
            Keys are parameter names from search_space, values are the
            optimal values for those parameters.

        Example:
            best_params = optimizer.optimize()
            # Returns: {"embedding_model": "all-mpnet-base-v2", "k_neighbors": 35}

            # Use best params to create final blocker
            blocker = VectorBlocker(
                embedding_model=best_params["embedding_model"],
                k_neighbors=best_params["k_neighbors"],
                ...
            )

        Note:
            This method logs optimization progress using the logging module.
            Set logging level to INFO to see trial results.
        """
        logger.info(
            "Starting optimization with %d trials (direction: %s)", self.n_trials, self.direction
        )

        # Create Optuna study
        study = self._create_study()

        # Set up callbacks
        callbacks = []
        if self.wandb_kwargs:
            wandb_callback = WeightsAndBiasesCallback(**self.wandb_kwargs)
            callbacks.append(wandb_callback)
            logger.info("wandb integration enabled for optimization")

        # Run optimization
        study.optimize(self._objective_wrapper, n_trials=self.n_trials, callbacks=callbacks)

        # Get best parameters
        best_params = study.best_trial.params

        logger.info("Optimization complete. Best parameters: %s", best_params)
        logger.info("Best value: %.4f", study.best_trial.value)

        return best_params

    def _create_study(self) -> optuna.Study:
        """Create Optuna study with specified direction.

        Returns:
            Optuna Study object configured for optimization direction.
        """
        return optuna.create_study(direction=self.direction)

    def _objective_wrapper(self, trial: optuna.Trial) -> float:
        """Wrapper for objective function that suggests parameters from trial.

        This method is called by Optuna for each trial. It:
        1. Suggests hyperparameters based on search space
        2. Calls the user's objective function with suggested params
        3. Returns the metric value to Optuna

        Args:
            trial: Optuna Trial object for suggesting parameters

        Returns:
            Metric value from objective function (float)

        Note:
            This method automatically detects parameter types:
            - list = categorical parameter
            - tuple = integer range parameter
        """
        # Suggest parameters based on search space
        params: dict[str, int | str] = {}
        for param_name, param_spec in self.search_space.items():
            if isinstance(param_spec, list):
                # Categorical parameter
                value = trial.suggest_categorical(param_name, param_spec)
                if value is not None:  # pragma: no cover
                    params[param_name] = value
            elif isinstance(param_spec, tuple) and len(param_spec) == 2:
                # Integer range parameter
                params[param_name] = trial.suggest_int(param_name, param_spec[0], param_spec[1])
            else:
                raise ValueError(
                    f"Invalid parameter specification for '{param_name}': {param_spec}. "
                    "Expected list (categorical) or tuple of (min, max) (integer)."
                )

        logger.debug("Trial parameters: %s", params)

        # Call objective function
        result = self.objective_fn(params)

        # Handle dict return (log all metrics, return primary)
        if isinstance(result, dict):
            # Validate primary metric exists
            if self.primary_metric not in result:
                raise ValueError(
                    f"Primary metric '{self.primary_metric}' not found in result. "
                    f"Available: {list(result.keys())}"
                )

            # Extract primary metric value
            primary_value = result[self.primary_metric]

            # Log all other metrics as user attributes (for wandb)
            for key, value in result.items():
                if key != self.primary_metric:
                    trial.set_user_attr(key, value)

            logger.debug(
                "Trial value (primary_metric=%s): %.4f", self.primary_metric, primary_value
            )
            return primary_value

        # Handle float return (backwards compatibility)
        logger.debug("Trial value: %.4f", result)
        return result
