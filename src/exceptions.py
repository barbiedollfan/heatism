# Universal error class
class SimulationError(Exception):
    pass

# Recoverable or non-fatal errors
class InputError(SimulationError):
    pass

class UninitializedError(InputError):
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
