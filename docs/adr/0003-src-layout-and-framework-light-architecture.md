# ADR 0003: Src Layout and Framework-Light Architecture

## Decision

Use an installable `src/kawaneen` layout. Keep the CLI on stdlib `argparse`, settings in Pydantic Settings, and logging in structlog integrated with standard logging.

## Context

The project needs import behavior that matches installed behavior and a small foundation that does not commit prematurely to an application framework.

## Consequences

Import paths are protected from accidental working-directory imports. The CLI stays dependency-light, while configuration and logging have clear seams for later modules. No web framework or UI is required in Phase 0.
