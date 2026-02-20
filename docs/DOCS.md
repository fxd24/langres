# Documentation Implementation Plan
 TODO: see new docs framework: https://github.com/zensical/zensical (from Material for MkDocs creator)
## Stack Decision

**Chosen Stack**: MkDocs + Material Theme + mkdocstrings + GitHub Pages

### Why This Stack?

1. **Modern, polished UI** - Material theme matches quality of FastAPI, Pydantic
2. **Auto-generates API docs** - mkdocstrings extracts from Google-style docstrings
3. **Narrative docs support** - Can publish existing docs (POC.md, PROJECT_OVERVIEW.md, etc.)
4. **LLM-optimized** - Clean structure, consistent terminology, self-contained sections
5. **Easy deployment** - GitHub Actions auto-deploys on push
6. **Future-proof** - Room to add tutorials, guides, examples as project grows

### Key Features

- **Automatic API reference** from docstrings
- **Dark mode** built-in
- **Fast search** with instant results
- **Mobile-responsive** design
- **Code copy buttons** on all examples
- **Version warnings** (can add later when needed)

## Implementation Roadmap

### Phase 1: Basic Setup (1-2 hours)

#### Step 1.1: Install Dependencies

```bash
uv add --dev mkdocs-material "mkdocstrings[python]"
```

#### Step 1.2: Create `mkdocs.yml`

Create the main configuration file at the repository root. See the Configuration section below for the complete config.

#### Step 1.3: Create Documentation Structure

```bash
mkdir -p docs/concepts docs/guides docs/api/core docs/development
```

File structure:
```
docs/
├── index.md                    # Homepage (to be created)
├── getting-started.md          # Quick start guide (to be created)
├── concepts/
│   ├── architecture.md         # Move from PROJECT_OVERVIEW.md
│   ├── data-contracts.md       # Move from TECHNICAL_OVERVIEW.md
│   └── use-cases.md            # Move from USE_CASES.md
├── guides/
│   ├── deduplication.md        # Tutorial (future)
│   ├── entity-linking.md       # Tutorial (future)
│   └── custom-modules.md       # Advanced guide (future)
├── api/
│   └── core/
│       ├── module.md           # API ref (to be created)
│       └── models.md           # API ref (to be created)
└── development/
    ├── poc.md                  # Move from POC.md
    └── contributing.md         # Create from CONTRIBUTING.md if exists
```

#### Step 1.4: Create Homepage

Create `docs/index.md` with project introduction, features, quick start. See Template section below.

#### Step 1.5: Test Locally

```bash
uv run mkdocs serve
```

Visit http://127.0.0.1:8000/ to preview.

### Phase 2: Move Existing Docs (1 hour)

#### Step 2.1: Move and Adapt Existing Documentation

```bash
# Move existing docs into structure
cp docs/POC.md docs/development/poc.md
cp docs/PROJECT_OVERVIEW.md docs/concepts/architecture.md
cp docs/TECHNICAL_OVERVIEW.md docs/concepts/data-contracts.md
cp docs/USE_CASES.md docs/concepts/use-cases.md
```

Review each file for:
- Fix any relative links
- Ensure heading hierarchy is correct (no skipped levels)
- Add introductory context to each section (LLM-friendly)

#### Step 2.2: Create API Reference Pages

Create minimal API reference pages that use mkdocstrings auto-generation:

**docs/api/core/module.md**:
```markdown
# Module

::: langres.core.Module
    options:
      show_root_heading: true
      show_source: true
      members_order: source
      show_signature_annotations: true
```

**docs/api/core/models.md**:
```markdown
# Data Models

::: langres.core.models
    options:
      show_root_heading: true
      show_source: true
      members_order: source
      show_signature_annotations: true
```

### Phase 3: GitHub Actions Deployment (30 min)

#### Step 3.1: Create Workflow File

Create `.github/workflows/docs.yml` (see Configuration section below).

#### Step 3.2: Enable GitHub Pages

1. Go to repository Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` / `root`
4. Save

#### Step 3.3: Push and Deploy

```bash
git add mkdocs.yml docs/ .github/workflows/docs.yml
git commit -m "docs: Set up MkDocs with Material theme"
git push
```

After GitHub Actions completes, docs will be at: `https://<username>.github.io/langres/`

### Phase 4: Enhance Docstrings (Ongoing)

#### Step 4.1: Docstring Guidelines

All public APIs must use **Google-style docstrings**:

```python
def compute_similarity(left: Entity, right: Entity, threshold: float = 0.5) -> float:
    """Compute similarity between two entities.

    This function compares two entities and returns a calibrated similarity score.
    The comparison uses the configured matching strategy.

    Args:
        left: The left entity to compare
        right: The right entity to compare
        threshold: Minimum similarity threshold. Must be between 0 and 1.
            Defaults to 0.5.

    Returns:
        Similarity score between 0 (completely different) and 1 (identical match).

    Raises:
        ValueError: If threshold is not between 0 and 1.

    Example:
        ```python
        from langres.core import Module
        from langres.core.models import Entity

        left = Entity(id="1", attributes={"name": "Acme Corp"})
        right = Entity(id="2", attributes={"name": "ACME Corporation"})

        module = Module()
        score = module.compute_similarity(left, right, threshold=0.7)
        print(f"Similarity: {score}")  # Output: Similarity: 0.85
        ```

    Note:
        This method is called internally by `forward()` for each candidate pair.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    ...
```

**Required sections**:
- Summary (first line + optional extended description)
- `Args:` - All parameters with types and descriptions
- `Returns:` - What is returned
- `Raises:` - Exceptions that can be raised (if any)
- `Example:` - At least one working code example

**Optional but recommended**:
- `Note:` - Important information
- `Warning:` - Caveats or gotchas
- `See Also:` - Related functions/classes

#### Step 4.2: Add Examples to All Public Methods

Every public method should have an `Example:` section showing:
- Imports needed
- Minimal working example
- Expected output (as comment)

#### Step 4.3: Verify Documentation Renders

After updating docstrings:
```bash
uv run mkdocs serve
```

Check that:
- Type hints render correctly
- Examples are syntax-highlighted
- Links work (mkdocstrings auto-links to other classes)

### Phase 5: Polish (Before V1)

#### Step 5.1: Create Tutorials

Add hands-on tutorials in `docs/guides/`:
- `deduplication.md` - End-to-end deduplication example
- `entity-linking.md` - Entity linking walkthrough
- `custom-modules.md` - Building custom matching logic

#### Step 5.2: Create Getting Started Guide

`docs/getting-started.md` should include:
- Installation instructions
- 5-minute quick start
- "Hello World" equivalent for entity resolution
- Next steps (link to tutorials)

#### Step 5.3: Add Badges

Consider adding to `docs/index.md`:
- Build status badge
- Test coverage badge
- PyPI version badge (when published)
- Python version badge

#### Step 5.4: Optional Enhancements

Consider adding these MkDocs plugins:
- `mkdocs-jupyter` - Render Jupyter notebooks as docs
- `mkdocs-git-revision-date-locator-plugin` - Show "Last updated" dates
- `mkdocs-minify-plugin` - Minify HTML/CSS/JS
- Social cards (Material theme feature) - Auto-generate preview images

## Configuration Files

### `mkdocs.yml` (Repository Root)

```yaml
site_name: langres
site_url: https://yourusername.github.io/langres  # Update with your GitHub username
site_description: Composable entity resolution framework for Python
site_author: Your Name  # Update with your name

repo_name: yourusername/langres  # Update with your GitHub username
repo_url: https://github.com/yourusername/langres  # Update with your GitHub username

theme:
  name: material
  palette:
    # Light mode
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    # Dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.instant       # Fast page loads
    - navigation.tracking      # URL updates on scroll
    - navigation.tabs          # Top-level tabs
    - navigation.sections      # Collapsible sections
    - navigation.expand        # Expanded sidebar
    - navigation.top           # Back to top button
    - search.suggest           # Search suggestions
    - search.highlight         # Highlight search terms
    - content.code.copy        # Copy button on code blocks
    - content.code.annotate    # Code annotations

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]  # Where to find your Python code
          options:
            docstring_style: google
            show_source: true
            show_root_heading: true
            show_category_heading: true
            members_order: source
            separate_signature: true
            show_signature_annotations: true
            line_length: 80

markdown_extensions:
  # Code highlighting
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.superfences

  # Tabbed content
  - pymdownx.tabbed:
      alternate_style: true

  # Admonitions (callout boxes)
  - admonition
  - pymdownx.details

  # Tables
  - tables

  # Task lists
  - pymdownx.tasklist:
      custom_checkbox: true

  # Better lists
  - def_list

  # Footnotes
  - footnotes

  # Attributes on elements
  - attr_list
  - md_in_html

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Concepts:
      - Architecture: concepts/architecture.md
      - Data Contracts: concepts/data-contracts.md
      - Use Cases: concepts/use-cases.md
  - Guides:
      - Deduplication: guides/deduplication.md
      - Entity Linking: guides/entity-linking.md
      - Custom Modules: guides/custom-modules.md
  - API Reference:
      - Core:
          - Module: api/core/module.md
          - Models: api/core/models.md
      # Add more as you build
  - Development:
      - POC Plan: development/poc.md
      - Contributing: development/contributing.md
```

### `.github/workflows/docs.yml`

```yaml
name: Deploy Documentation

on:
  push:
    branches:
      - main
      - poc/core-foundation  # Add any other branches you want docs for
  pull_request:

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: |
          uv pip install --system mkdocs-material "mkdocstrings[python]"

      - name: Build documentation
        run: mkdocs build

      - name: Deploy to GitHub Pages
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        run: mkdocs gh-deploy --force
```

### `docs/index.md` Template

```markdown
# langres

**Composable entity resolution framework for Python**

langres provides a flexible, optimizable approach to entity resolution with a clean two-layer API:
- **High-level tasks** for common use cases (deduplication, entity linking)
- **Low-level primitives** for custom pipelines (modules, blockers, clusterers)

## Key Features

- **Composable Architecture**: Mix and match components (blockers, modules, optimizers)
- **Multiple Approaches**: Classical string matching, semantic embeddings, or LLM-based
- **Full Observability**: Every decision includes provenance and reasoning
- **Optimizable**: Built-in support for hyperparameter tuning (Optuna) and prompt optimization (DSPy)
- **Type-Safe**: Comprehensive type hints and Pydantic validation throughout

## Status

⚠️ **Early Development (POC Stage)**: This project is currently validating core architecture through a proof-of-concept. See [POC Plan](development/poc.md) for details.

## Quick Start

```python
# Coming soon - install from PyPI
# pip install langres

from langres.core import Module
from langres.core.models import Entity

# Define entities
entities = [
    Entity(id="1", attributes={"name": "Acme Corp", "location": "NYC"}),
    Entity(id="2", attributes={"name": "ACME Corporation", "location": "New York"}),
]

# Create and run module
module = Module()
results = module.run(entities)
```

## Use Cases

langres is designed for:

- **Deduplication**: Find duplicate records within a single dataset
- **Entity Linking**: Match records across different datasets
- **Entity Resolution**: Cluster related entities and create canonical representations

See [Use Cases](concepts/use-cases.md) for detailed taxonomy.

## Documentation Structure

- **[Getting Started](getting-started.md)**: Installation and first steps
- **[Concepts](concepts/architecture.md)**: Architecture and design philosophy
- **[Guides](guides/deduplication.md)**: Hands-on tutorials and examples
- **[API Reference](api/core/module.md)**: Complete API documentation
- **[Development](development/poc.md)**: Contributing and development roadmap

## Why langres?

Existing entity resolution tools often force you into rigid workflows. langres provides:

1. **Flexibility**: Compose custom pipelines from reusable primitives
2. **Transparency**: Full provenance for every matching decision
3. **Optimization**: Built-in tuning for both traditional ML and LLM approaches
4. **Modern Python**: Type hints, Pydantic, async support (coming)

## Project Status

Current focus: Building production-quality `langres.core` primitives with 100% test coverage.

See the [POC Plan](development/poc.md) for:
- Validation hypothesis
- Success criteria (BCubed F1 ≥ 0.85)
- Three progressive approaches being tested
- Go/No-Go decision criteria

## License

[Your License Here]

## Citation

[Add if academic context]
```

## LLM-Era Best Practices

### Critical for AI Consumption

1. **Consistent Terminology**
   - ✅ Pick one term and use everywhere: `Module` (not "Flow", "Matcher", "Comparator")
   - ✅ Define terms on first use
   - ✅ Use the same term in docs, code, and docstrings

2. **Self-Contained Sections**
   - ❌ "As mentioned above, configure the blocker..."
   - ✅ "Configure the blocker (which generates candidate pairs) by..."

3. **Proper Code Fencing**
   - ❌ Use the `langres.Module` class and call `run()`
   - ✅ Use code blocks with syntax highlighting (see examples above)

4. **Clear Hierarchy**
   - ✅ No skipped heading levels (H1 → H2 → H3, never H1 → H3)
   - ✅ Logical structure: Concept → Example → Reference

5. **Examples in Every Docstring**
   - ✅ Every public method needs an `Example:` section
   - ✅ Examples should be runnable (imports + code + expected output)

6. **Type Hints Everywhere**
   - ✅ You're already doing this! Keep it up.
   - ✅ mkdocstrings will render these beautifully

## Maintenance

### Regular Tasks

- **After adding new module**: Create API reference page in `docs/api/`
- **After adding feature**: Add example to relevant guide
- **Before release**: Update getting-started guide
- **When API changes**: Update docstrings immediately (they're the source of truth)

### Monitoring

Check documentation build status:
- GitHub Actions tab shows build success/failure
- Visit docs site to verify rendering
- Test search functionality periodically

### Versioning (Future)

When ready for versioning (post-V1):
- Consider migrating to Read the Docs for automatic version switching
- Or use `mike` plugin with MkDocs for multi-version support

## Troubleshooting

### Common Issues

**Build fails with "No module named 'langres'"**
- Ensure `paths: [src]` is in mkdocs.yml
- Verify package is importable: `uv run python -c "import langres.core"`

**Docstrings not rendering**
- Check Google-style formatting is correct
- Verify indentation (use spaces, not tabs)
- Check `show_source: true` is in mkdocstrings config

**Search not working**
- Material theme search requires JavaScript enabled
- Check browser console for errors
- Ensure `- search` plugin is in mkdocs.yml

**Links broken after deployment**
- Use relative links: `[text](../other-page.md)`
- Or absolute from root: `[text](/guides/tutorial.md)`
- Test with `mkdocs serve` before deploying

## Resources

- **MkDocs Documentation**: https://www.mkdocs.org/
- **Material Theme**: https://squidfunk.github.io/mkdocs-material/
- **mkdocstrings**: https://mkdocstrings.github.io/
- **Google Docstring Guide**: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
- **FastAPI Docs** (reference example): https://github.com/fastapi/fastapi/tree/master/docs

## Next Steps

1. Complete Phase 1 setup (2 hours)
2. Test locally with `uv run mkdocs serve`
3. Deploy to GitHub Pages (Phase 3)
4. Gradually enhance docstrings (Phase 4)
5. Add tutorials before V1 release (Phase 5)

## Success Criteria

Documentation is ready for V1 when:
- ✅ All public APIs have Google-style docstrings with examples
- ✅ Getting Started guide exists and is tested
- ✅ At least 2 tutorials exist (deduplication + entity linking)
- ✅ API reference auto-generates correctly
- ✅ Site deploys automatically on push
- ✅ Search works and finds relevant content
- ✅ Mobile view is tested and works well
- ✅ All links are verified (no 404s)
