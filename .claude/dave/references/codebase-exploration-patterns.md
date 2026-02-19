# Codebase Exploration Patterns

Reusable search patterns for finding services, gateways, repositories, models, config, and data flow in the codebase.

## Finding Service Patterns
```bash
# Find all orchestrating services
Glob: src/services/**/*service*.py

# Find how a specific service is structured
Read: src/services/{name}/service.py

# Find who calls a service
Grep: from src.services.{name} import
Grep: {ServiceClass}(
```

## Finding Gateway Patterns
```bash
# Find all gateways
Glob: src/clients/**/*gateway*.py

# Find how gateways are instantiated
Grep: get_{name}_gateway
Grep: {GatewayClass}(

# Find gateway configuration
Grep: class {GatewayClass}
```

## Finding Repository Patterns
```bash
# Find all repositories
Glob: src/infrastructure/repositories/**/*.py

# Find base repository pattern
Read: src/infrastructure/repositories/base.py (or similar)

# Find how repos are used in services
Grep: Repository(session)
```

## Finding Domain Models
```bash
# Find all domain models
Glob: src/domain/models/**/*.py

# Find database models
Read: src/infrastructure/database/models/db_models.py

# Find how models are used
Grep: class {ModelName}
```

## Finding Configuration Patterns
```bash
# Find config module
Read: src/config.py

# Find environment variable usage
Grep: os.environ
Grep: settings\.
```

## Tracing Data Flow
```bash
# Start from pipeline/entry point
Grep: def {entry_function}
# Follow the call chain through each layer
# Service -> Gateway -> External
# Service -> Repository -> Database
```
