"""Tests for langres.core.optimizers.blocker_optimizer module."""

from unittest.mock import MagicMock, call, patch

import pytest

from langres.core.optimizers.blocker_optimizer import BlockerOptimizer


class TestBlockerOptimizer:
    """Tests for BlockerOptimizer class."""

    def test_blocker_optimizer_initialization(self):
        """Test BlockerOptimizer initializes with correct parameters."""
        objective = lambda trial, params: {"value": params["k_neighbors"] * 0.1}
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
        def objective(trial, params: dict) -> dict:
            score = params["k_neighbors"] * 0.01
            if params["embedding_model"] == "model-a":
                score += 0.5
            return {"value": score}

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

        objective = lambda trial, params: {"value": 1.0}
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
        objective = lambda trial, params: {"value": 1.0}
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
        objective = lambda trial, params: {"value": params["k_neighbors"] * 0.1}
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
        objective = lambda trial, params: {"value": 0.85}
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
        objective = lambda trial, params: {"value": 0.15}
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
        objective = lambda trial, params: {"value": 0.85}
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
        objective = lambda trial, params: {"value": 0.85}
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

        def objective(trial, params: dict) -> dict:
            received_params.append(params.copy())
            return {"value": 0.85}

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
        objective = lambda trial, params: {"value": 0.85}
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
        objective = lambda trial, params: {"value": 0.85}

        with pytest.raises(ValueError, match="search_space cannot be empty"):
            BlockerOptimizer(
                objective_fn=objective,
                search_space={},
                direction="maximize",
                n_trials=5,
            )

    def test_blocker_optimizer_raises_error_for_invalid_parameter_spec(self):
        """Test that BlockerOptimizer raises ValueError for invalid parameter specification."""
        objective = lambda trial, params: {"value": 0.85}
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

    def test_blocker_optimizer_dict_return_with_valid_primary_metric(self):
        """Test objective function returning dict with valid primary_metric."""

        def objective(trial, params: dict) -> dict:
            return {
                "bcubed_f1": 0.85,
                "bcubed_precision": 0.90,
                "pairwise_f1": 0.80,
                "cost_usd": 0.05,
            }

        search_space = {"k_neighbors": (5, 50)}

        with patch("langres.core.optimizers.blocker_optimizer.optuna"):
            optimizer = BlockerOptimizer(
                objective_fn=objective,
                search_space=search_space,
                primary_metric="bcubed_f1",
                n_trials=1,
            )

            mock_trial = MagicMock()
            mock_trial.suggest_int.return_value = 10

            result = optimizer._objective_wrapper(mock_trial)

            # Verify primary metric returned
            assert result == 0.85

            # Verify other metrics logged as user_attrs
            expected_calls = [
                call("bcubed_precision", 0.90),
                call("pairwise_f1", 0.80),
                call("cost_usd", 0.05),
            ]
            mock_trial.set_user_attr.assert_has_calls(expected_calls, any_order=True)

            # Verify primary metric NOT logged as user_attr
            for c in mock_trial.set_user_attr.call_args_list:
                assert c[0][0] != "bcubed_f1"

    def test_blocker_optimizer_dict_return_missing_primary_metric(self):
        """Test error when objective returns dict without primary_metric."""

        def objective(trial, params: dict) -> dict:
            return {
                "bcubed_precision": 0.90,
                "pairwise_f1": 0.80,
            }

        search_space = {"k_neighbors": (5, 50)}

        with patch("langres.core.optimizers.blocker_optimizer.optuna"):
            optimizer = BlockerOptimizer(
                objective_fn=objective,
                search_space=search_space,
                primary_metric="bcubed_f1",  # Missing from result!
                n_trials=1,
            )

            mock_trial = MagicMock()
            mock_trial.suggest_int.return_value = 10

            with pytest.raises(
                ValueError,
                match="Primary metric 'bcubed_f1' not found in result",
            ):
                optimizer._objective_wrapper(mock_trial)

    def test_blocker_optimizer_float_return_raises_error(self):
        """Test that objective returning float raises TypeError."""

        def objective(trial, params: dict) -> float:
            return 0.85

        search_space = {"k_neighbors": (5, 50)}

        with patch("langres.core.optimizers.blocker_optimizer.optuna"):
            optimizer = BlockerOptimizer(
                objective_fn=objective,
                search_space=search_space,
                primary_metric="value",
                n_trials=1,
            )

            mock_trial = MagicMock()
            mock_trial.suggest_int.return_value = 10

            with pytest.raises(TypeError, match="objective_fn must return dict"):
                optimizer._objective_wrapper(mock_trial)

    def test_blocker_optimizer_default_primary_metric(self):
        """Test default primary_metric value is 'value'."""

        def objective(trial, params: dict) -> dict:
            return {
                "value": 0.85,  # Default key
                "precision": 0.90,
                "cost": 0.05,
            }

        search_space = {"k_neighbors": (5, 50)}

        with patch("langres.core.optimizers.blocker_optimizer.optuna"):
            # Don't specify primary_metric - should default to "value"
            optimizer = BlockerOptimizer(
                objective_fn=objective,
                search_space=search_space,
                n_trials=1,
            )

            assert optimizer.primary_metric == "value"

            mock_trial = MagicMock()
            mock_trial.suggest_int.return_value = 10

            result = optimizer._objective_wrapper(mock_trial)

            # Verify primary metric ("value") returned
            assert result == 0.85

            # Verify other metrics logged
            expected_calls = [
                call("precision", 0.90),
                call("cost", 0.05),
            ]
            mock_trial.set_user_attr.assert_has_calls(expected_calls, any_order=True)

    def test_objective_fn_receives_trial(self):
        """Objective function must receive trial as first parameter."""
        import optuna

        trial_numbers = []

        def objective(trial: optuna.Trial, params: dict) -> dict:
            trial_numbers.append(trial.number)
            return {"bcubed_f1": 0.85}

        search_space = {"k_neighbors": (5, 50)}

        optimizer = BlockerOptimizer(
            objective_fn=objective,
            search_space=search_space,
            primary_metric="bcubed_f1",
            n_trials=3,
        )

        optimizer.optimize()
        assert trial_numbers == [0, 1, 2]

    def test_objective_can_use_trial_api(self):
        """Objective can call trial.set_user_attr() and other trial methods."""
        import optuna

        def objective(trial: optuna.Trial, params: dict) -> dict:
            trial.set_user_attr("custom_data", "test_value")
            return {"bcubed_f1": 0.85}

        search_space = {"k_neighbors": (5, 50)}

        with patch("langres.core.optimizers.blocker_optimizer.optuna") as mock_optuna:
            mock_study = MagicMock()
            mock_trial = MagicMock()
            mock_trial.params = {"k_neighbors": 10}
            mock_trial.value = 0.85
            mock_trial.user_attrs = {"custom_data": "test_value"}
            mock_study.best_trial = mock_trial
            mock_optuna.create_study.return_value = mock_study

            optimizer = BlockerOptimizer(
                objective_fn=objective,
                search_space=search_space,
                primary_metric="bcubed_f1",
                n_trials=1,
            )

            optimizer.optimize()

            # Verify trial.set_user_attr was called during objective execution
            # (this will be called by _objective_wrapper when it passes trial to objective)
            assert mock_study.best_trial.user_attrs["custom_data"] == "test_value"

    def test_old_signature_raises_error(self):
        """Old objective signature (without trial) raises TypeError."""

        def old_objective(params: dict) -> dict:
            return {"bcubed_f1": 0.85}

        search_space = {"k_neighbors": (5, 50)}

        optimizer = BlockerOptimizer(
            objective_fn=old_objective,
            search_space=search_space,
            primary_metric="bcubed_f1",
        )

        with pytest.raises(TypeError):
            optimizer.optimize()
