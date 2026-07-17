"""``langres.training``: fitting and calibrating a matcher.

What produces a tuned model -- not entity-resolution modelling itself. The
distinction is the same one that moved :mod:`langres.report` out of
``langres.core``: ``core`` is the contracts floor a pipeline is *written
against*; how you *arrive at* a fitted matcher is a separate concern with a
separate dependency stack.

The five modules here are that concern:

===========================  ==========================================
module                       what it owns
===========================  ==========================================
``finetune``                 QLoRA/LoRA training (``run_finetune``,
                             ``QLoRA``, ``LabeledCandidate``)
``calibration``              ``derive_threshold`` + the Platt/isotonic
                             ``Calibrator``
``fit_report``               the ``FitReport`` fit digest
``methods_prompt``           ``Bootstrap`` / ``MIPRO`` / ``GEPA``
``methods_calibrate``        ``Platt`` / ``Isotonic``
===========================  ==========================================

**Layering.** Every module here imports *downward* into ``core`` (
``core.methods_api``, ``core.model_ref``, ``core.registry``, ``core.harvest``,
``core.metrics``) and none of them import the ``langres.core`` facade, so
``training -> core`` is one-way at the module level.

``core -> training`` is **not** zero, and that is by design rather than
oversight -- see ``tests/test_import_tangle.py`` for the ratchet that measures
it. Three things import back up:

* ``core/_exports/_training.py`` re-exports ``FitReport`` / ``Bootstrap`` /
  ``MIPRO`` / ``GEPA`` / ``Platt`` / ``Isotonic`` / ``Calibrator``, because those
  names are part of ``langres.core.__all__`` and that public surface is a
  compatibility contract (``tests/test_public_surface_dod.py`` pins its size).
* ``core/resolver.py`` -- ``ERModel.fit()`` dispatches into ``finetune`` and
  ``calibration`` via **function-local** imports (the existing idiom for a heavy
  path), plus one toplevel ``fit_report`` import for the ``fit_report_``
  attribute's type.
* ``core/_model_state.py`` holds ``FitReport`` on the serialized model state.

None of those close a cycle: the modules ``training`` imports (``harvest``,
``metrics``, ``methods_api``, ``model_ref``, ``registry``) do not import the
``core`` facade back.

**Dependency weight.** This package is where the two heaviest optional stacks
live, and both stay lazy:

* ``calibration`` imports scikit-learn **at module scope** (the ``[trained]``
  extra), so ``Calibrator`` / ``derive_threshold`` are reached through
  ``LAZY_SYMBOLS`` in the export fragments -- never eagerly.
* ``finetune`` imports peft/trl/bitsandbytes/torch **inside**
  ``QLoRATrainer.train`` (the ``[finetune]`` extra), so importing the module --
  and therefore ``QLoRA`` / ``run_finetune`` -- costs nothing.

``tests/test_import_budget.py`` is the gate on both: a bare ``import langres``
must pull neither.

**This ``__init__`` is deliberately empty of imports**, the same choice
:mod:`langres.report` documents. Consumers reach a module directly
(``from langres.training.finetune import run_finetune``), and an import-free
package module keeps that free: re-exporting ``Calibrator`` here would drag
scikit-learn into every ``import langres.training.finetune``. The public homes
for these names are the ``langres`` and ``langres.core`` namespaces, both of
which resolve the heavy ones lazily.
"""
