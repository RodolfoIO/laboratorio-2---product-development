class DataContractError(ValueError):
    """A request can be understood, but violates a documented data policy."""


class DependencyUnavailableError(RuntimeError):
    """An optional or required platform dependency is currently unavailable."""


class ModelUnavailableError(RuntimeError):
    """No forecasting model is currently available for serving."""


class OrchestrationError(RuntimeError):
    """A retraining run could not be started or queried."""
