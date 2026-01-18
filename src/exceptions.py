# Recoverable or non-fatal errors
class InputError(Exception):
    pass


class UninitializedError(InputError):
    ...


class ParameterError(InputError):
    pass


class IncompatibleTypeError(InputError):
    pass


# Nonrecoverable or fatal errors
class InitializationError(Exception):
    pass


class JsonFileError(InitializationError):
    pass
