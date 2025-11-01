"""Tests for langres.core.optimizers.blocker_optimizer module."""

from unittest.mock import MagicMock, call, patch

import pytest

from langres.core.optimizers.blocker_optimizer import BlockerOptimizer


class TestBlockerOptimizer:
    """Tests for BlockerOptimizer class."""

    def test_blocker_optimizer_initialization(self):
        """Test BlockerOptimizer initializes with correct parameters."""
        objective = lambda params: params["k_neighbors"] * 0.1
        search_space = {
            "embedding_model": ["model-a", "model-b"],
            "k_neighbors": (5, 50),
        }

        optimizer = BlockerOptimizer(
            objective_fn=objective,
            search_space=search_space,
            direction="maximize",
            n_trials=10,
        )

        assert optimizer.objective_fn is objective
        assert optimizer.search_space == search_space
        assert optimizer.direction == "maximize"
        assert optimizer.n_trials == 10

    def test_blocker_optimizer_optimize_returns_best_params(self):
        """Test optimize() returns best parameters after trials."""

        # Simple objective: prefer model-a and higher k_neighbors
        def objective(params: dict) -> float:
            score = params["k_neighbors"] * 0.01
            if params["embedding_model"] == "model-a":
                score += 0.5
            return score

        search_space = {
            "embedding_model": ["model-a", "model-b"],
            "k_neighbors": (5, 50),
        }

        with patch("langres.core.optimizers.blocker_optimizer.optuna"):
            optimizer = BlockerOptimizer(
                objective_fn=objective,
                search_space=search_space,
                direction="maximize",
                n_trials=5,
            )

            # Mock optuna study
            mock_study = MagicMock()
            mock_trial = MagicMock()
            mock_trial.params = {"embedding_model": "model-a", "k_neighbors": 50}
            mock_study.best_trial = mock_trial

            with patch.object(optimizer, "_create_study", return_value=mock_study):
                best_params = optimizer.optimize()

                assert best_params == {"embedding_model": "model-a", "k_neighbors": 50}
                mock_study.optimize.assert_called_once()

    def test_blocker_optimizer_categorical_parameter(self):
        """Test optimizer suggests categorical parameters correctly."""

        objective = lambda params: 1.0
        search_space = {"embedding_model": ["model-a", "model-b", "model-c"]}

        with patch("langres.core.optimizers.blocker_optimizer.optuna") as mock_optuna:
            optimizer = BlockerOptimizer(
                objective_fn=objective, search_space=search_space, n_trials=1
            )

            # Create mock trial
            mock_trial = MagicMock()
            mock_trial.suggest_categorical.return_value = "model-a"

            # Call objective wrapper
            result = optimizer._objective_wrapper(mock_trial)

            # Verify categorical suggestion called
            mock_trial.suggest_categorical.assert_called_once_with(
                "embedding_model", ["model-a", "model-b", "model-c"]
            )

    def test_blocker_optimizer_integer_parameter(self):
        """Test optimizer suggests integer parameters correctly."""
        objective = lambda params: 1.0
        search_space = {"k_neighbors": (5, 50)}

        with patch("langres.core.optimizers.blocker_optimizer.optuna"):
            optimizer = BlockerOptimizer(
                objective_fn=objective, search_space=search_space, n_trials=1
            )

            # Create mock trial
            mock_trial = MagicMock()
            mock_trial.suggest_int.return_value = 20

            # Call objective wrapper
            result = optimizer._objective_wrapper(mock_trial)

            # Verify integer suggestion called
            mock_trial.suggest_int.assert_called_once_with("k_neighbors", 5, 50)

    def test_blocker_optimizer_mixed_parameters(self):
        """Test optimizer with both categorical and integer parameters."""
        objective = lambda params: params["k_neighbors"] * 0.1
        search_space = {
            "embedding_model": ["model-a", "model-b"],
            "k_neighbors": (5, 50),
        }

        with patch("langres.core.optimizers.blocker_optimizer.optuna"):
            optimizer = BlockerOptimizer(
                objective_fn=objective, search_space=search_space, n_trials=1
            )

            mock_trial = MagicMock()
            mock_trial.suggest_categorical.return_value = "model-a"
            mock_trial.suggest_int.return_value = 30

            # Call objective wrapper
            result = optimizer._objective_wrapper(mock_trial)

            # Verify both suggestions called
            mock_trial.suggest_categorical.assert_called_once()
            mock_trial.suggest_int.assert_called_once()

            # Verify objective called with correct params
            assert result == 3.0  # 30 * 0.1

    def test_blocker_optimizer_direction_maximize(self):
        """Test optimizer with direction='maximize'."""
        objective = lambda params: 0.85
        search_space = {"k_neighbors": (5, 50)}

        with patch("langres.core.optimizers.blocker_optimizer.optuna") as mock_optuna:
            mock_study = MagicMock()
            mock_optuna.create_study.return_value = mock_study

            optimizer = BlockerOptimizer(
                objective_fn=objective,
                search_space=search_space,
                direction="maximize",
                n_trials=5,
            )

            optimizer.optimize()

            # Verify create_study called with direction="maximize"
            mock_optuna.create_study.assert_called_once_with(direction="maximize")

    def test_blocker_optimizer_direction_minimize(self):
        """Test optimizer with direction='minimize'."""
        objective = lambda params: 0.15
        search_space = {"k_neighbors": (5, 50)}

        with patch("langres.core.optimizers.blocker_optimizer.optuna") as mock_optuna:
            mock_study = MagicMock()
            mock_optuna.create_study.return_value = mock_study

            optimizer = BlockerOptimizer(
                objective_fn=objective,
                search_space=search_space,
                direction="minimize",
                n_trials=5,
            )

            optimizer.optimize()

            # Verify create_study called with direction="minimize"
            mock_optuna.create_study.assert_called_once_with(direction="minimize")

    def test_blocker_optimizer_wandb_integration(self):
        """Test optimizer integrates with wandb callback."""
        objective = lambda params: 0.85
        search_space = {"k_neighbors": (5, 50)}

        wandb_kwargs = {"project": "test-project", "entity": "test-team"}

        with patch("langres.core.optimizers.blocker_optimizer.optuna") as mock_optuna:
            with patch(
                "langres.core.optimizers.blocker_optimizer.WeightsAndBiasesCallback"
            ) as mock_wandb_callback:
                mock_study = MagicMock()
                mock_optuna.create_study.return_value = mock_study
                mock_callback_instance = MagicMock()
                mock_wandb_callback.return_value = mock_callback_instance

                optimizer = BlockerOptimizer(
                    objective_fn=objective,
                    search_space=search_space,
                    n_trials=5,
                    wandb_kwargs=wandb_kwargs,
                )

                optimizer.optimize()

                # Verify WeightsAndBiasesCallback created with wandb_kwargs
                mock_wandb_callback.assert_called_once_with(**wandb_kwargs)

                # Verify optimize called with callback
                mock_study.optimize.assert_called_once_with(
                    optimizer._objective_wrapper, n_trials=5, callbacks=[mock_callback_instance]
                )

    def test_blocker_optimizer_without_wandb_kwargs(self):
        """Test optimizer without wandb integration."""
        objective = lambda params: 0.85
        search_space = {"k_neighbors": (5, 50)}

        with patch("langres.core.optimizers.blocker_optimizer.optuna") as mock_optuna:
            mock_study = MagicMock()
            mock_optuna.create_study.return_value = mock_study

            optimizer = BlockerOptimizer(
                objective_fn=objective,
                search_space=search_space,
                n_trials=5,
                wandb_kwargs=None,  # No wandb
            )

            optimizer.optimize()

            # Verify optimize called WITHOUT callback
            mock_study.optimize.assert_called_once_with(
                optimizer._objective_wrapper, n_trials=5, callbacks=[]
            )

    def test_blocker_optimizer_objective_called_with_params(self):
        """Test that objective function receives correct parameters."""
        received_params = []

        def objective(params: dict) -> float:
            received_params.append(params.copy())
            return 0.85

        search_space = {
            "embedding_model": ["model-a"],
            "k_neighbors": (10, 10),  # Fixed value for deterministic test
        }

        with patch("langres.core.optimizers.blocker_optimizer.optuna"):
            optimizer = BlockerOptimizer(
                objective_fn=objective, search_space=search_space, n_trials=1
            )

            mock_trial = MagicMock()
            mock_trial.suggest_categorical.return_value = "model-a"
            mock_trial.suggest_int.return_value = 10

            optimizer._objective_wrapper(mock_trial)

            # Verify objective received correct params
            assert len(received_params) == 1
            assert received_params[0] == {"embedding_model": "model-a", "k_neighbors": 10}

    def test_blocker_optimizer_raises_error_for_invalid_direction(self):
        """Test that BlockerOptimizer raises ValueError for invalid direction."""
        objective = lambda params: 0.85
        search_space = {"k_neighbors": (5, 50)}

        with pytest.raises(ValueError, match="direction must be 'maximize' or 'minimize'"):
            BlockerOptimizer(
                objective_fn=objective,
                search_space=search_space,
                direction="invalid",
                n_trials=5,
            )

    def test_blocker_optimizer_raises_error_for_empty_search_space(self):
        """Test that BlockerOptimizer raises ValueError for empty search_space."""
        objective = lambda params: 0.85

        with pytest.raises(ValueError, match="search_space cannot be empty"):
            BlockerOptimizer(
                objective_fn=objective,
                search_space={},
                direction="maximize",
                n_trials=5,
            )

    def test_blocker_optimizer_raises_error_for_invalid_parameter_spec(self):
        """Test that BlockerOptimizer raises ValueError for invalid parameter specification."""
        objective = lambda params: 0.85
        # Invalid parameter spec: neither list nor tuple
        search_space = {"k_neighbors": "invalid"}

        with patch("langres.core.optimizers.blocker_optimizer.optuna"):
            optimizer = BlockerOptimizer(
                objective_fn=objective,
                search_space=search_space,
                direction="maximize",
                n_trials=1,
            )

            mock_trial = MagicMock()

            with pytest.raises(ValueError, match="Invalid parameter specification"):
                optimizer._objective_wrapper(mock_trial)
