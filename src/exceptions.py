class SimulationError(Exception):
    pass

class ParameterError(SimulationError):
    pass

class TypeError(ParameterError):
    pass
