class TestRailError(Exception):
    """Base exception for all TestRail client errors."""


class AuthenticationError(TestRailError):
    """Raised when authentication with TestRail fails."""


class AuthorizationError(TestRailError):
    """Raised when the authenticated user lacks permission."""


class ResourceNotFoundError(TestRailError):
    """Raised when a requested TestRail resource does not exist."""


class ValidationError(TestRailError):
    """Raised when TestRail rejects the request due to invalid input."""


class RateLimitError(TestRailError):
    """Raised when the API rate limit has been exceeded."""


class ServerError(TestRailError):
    """Raised when TestRail returns a server-side error."""


class RequestError(TestRailError):
    """Raised for network or transport errors."""