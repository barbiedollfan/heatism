import os
import platform
import json
from exceptions import (
    InitializationError,
    JsonFileError
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


def load_json(path):
    try:
        with open(path, "r") as f:
            try:
                file_contents = json.load(f)
            except Exception as e:
                raise JsonFileError(
                    f"could not decode {path}: {e}."
                )
    except FileNotFoundError:
        raise JsonFileError(
            f"could not find file at {path}."
        )
    return file_contents


def get_default_params(path):
    try:
        defaults = load_json(path)
    except InitializationError:
        raise
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
        raise JsonFileError(
            f"could not decode default parameters: {e}."
        )
    return defaults


def generate_materials_list(path):
    try:
        material_dict = load_json(path)
    except InitializationError:
        raise
    material_list = material_dict.keys()
    sorted_material_list = sorted(material_list, key=lambda x: material_dict[x]["k"]/(material_dict[x]["p"] * material_dict[x]["c"]), reverse=True)
    info = "\n"
    for material in sorted_material_list:
        info += material.capitalize() + "\n"
    return info


def generate_defaults_info(path):
    try:
        defaults = load_json(path)
    except InitializationError:
        raise
    info = f"""
Material: {defaults["material"].capitalize()}
Points: {defaults["points"]}x{defaults["points"]}
Sidefaultse Length: {defaults["side_length"]}m
Thickness: {defaults["thickness"]}m
Function: {defaults["function"].capitalize()}
Time Step: {defaults["dt"]}s
Min Temp: {defaults["min_temp"]}K
Max Temp: {defaults["max_temp"]}K
    """
    return info


def generate_functions_list(path):
    try:
        functions_dict = load_json(path)
    except InitializationError:
        raise
    info = "\n"
    for function, description in functions_dict.items():
        info += f"{function.capitalize()} - {description.capitalize()}\n"
    return info
