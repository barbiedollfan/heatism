# Recoverable or non-fatal errors
class InputError(SimulationError):
    pass


class ParameterError(InputError):
    pass


class IncompatibleTypeError(InputError):
    pass


# Nonrecoverable or fatal errors
class InitializationError(SimulationError):
    pass


class MaterialsFileError(InitializationError):
    pass


class DefaultsFileError(InitializationError):
    pass
