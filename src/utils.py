import os
import platform
import json
from exceptions import (
    InitializationError,
    DefaultsFileError
)

def clear():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def convert_energy(energy):
    if energy < 1000:
        return (energy, "J")
    elif energy < 1000**2:
        return (energy / 1000, "kJ")
    elif energy < 1000**3:
        return (energy / (1000**2), "MJ")
    elif energy < 1000**4:
        return (energy / (1000**3), "GJ")


def total_energy(p, c, temp_field, dv):
    total = 0
    for row in temp_field:
        for temp in row:
            total += p * c * temp * dv
    return convert_energy(total)


def get_default_params(path):
    try:
        with open(path, "r") as f:
            try:
                defaults = json.load(f)
            except Exception as e:
                raise DefaultsFileError(
                    f"Could not decode default parameters ({path}): {e}."
                )
    except FileNotFoundError:
        raise DefaultsFileError(
            f"Could not find default parameters file ({path})."
        )
    try:
        material = defaults["material"]
        points = defaults["points"]
        side_length = defaults["side_length"]
        function = defaults["function"]
        dt = defaults["dt"]
        thickness = defaults["thickness"]
        new_min = defaults["min_temp"]
        new_max = defaults["max_temp"]
    except Exception as e:
        raise DefaultsFileError(
            f"Could not decode default parameters ({path}): {e}."
        )
    return defaults


def generate_defaults_info(path):
    try:
        d = get_default_params(path)
    except InitializationError:
        raise
    info = f"""
Material: {d["material"].capitalize()}
Points: {d["points"]}x{d["points"]}
Side Length: {d["side_length"]}m
Thickness: {d["thickness"]}m
Function: {d["function"].capitalize()}
Time Step: {d["dt"]}s
Min Temp: {d["min_temp"]}K
Max Temp: {d["max_temp"]}K
    """
    return info
