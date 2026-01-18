import matplotlib.pyplot as plt
import numpy as np
import random, json, sys, threading, os, argparse
from backwards_euler import (
    insert_matrix,
    gen_coeff_matrix,
    gen_known_vector,
    next_temps,
)
from exceptions import (
    InputError,
    UninitializedError,
    ParameterError,
    IncompatibleTypeError,
    InitializationError,
    JsonFileError
)
import scipy.sparse as spr
import scipy.sparse.linalg as spl
import utils as ut
from time import sleep
from pathlib import Path
import gen.initial_gen as gen

base_dir = Path(__file__).parent
config_dir = base_dir.parent / "configs"
gen_dir = base_dir / "gen"

DEFAULTS_PATH = config_dir / "defaults.json"
MATERIALS_PATH = config_dir / "materials.json"
FUNCTIONS_PATH = gen_dir / "functions.json"


class Plate:
    def __init__(self, initial_heat_map, points, side_length):
        self.heat_map = initial_heat_map.copy()
        self.initial_heat_map = initial_heat_map.copy()
        self.points = points
        self.side_length = side_length
        self.dr = side_length / (self.points - 1)

    def gen_solver(self, dt):
        self.coeff = self.diffusivity * dt / (self.dr**2)
        coeff_matrix = gen_coeff_matrix(
            self.points - 2, 1 + 4 * self.coeff, -self.coeff
        )
        coeff_matrix = spr.csc_matrix(coeff_matrix)
        self.solve = spl.factorized(coeff_matrix)

    def gen_material_properties(self, material):
        try:
            material_dict = ut.load_json(MATERIALS_PATH) 
        except JsonFileError:
            raise 

        try:
            properties = material_dict[material]
        except KeyError:
            raise ParameterError(f'could not get properties for "{material}".')

        try:
            k = properties["k"]
            p = properties["p"]
            c = properties["c"]
        except Exception as e:
            raise JsonFileError(f"could not decode material properties: {e}.")
        self.c = c
        self.p = p
        self.diffusivity = k / (p * c)

    def update(self):
        t = gen_known_vector(self.heat_map, self.coeff)
        new_temps_vec = self.solve(t)
        new_temps_matrix = new_temps_vec.reshape(self.points - 2, self.points - 2)
        self.heat_map = insert_matrix(new_temps_matrix, self.heat_map, 1, 1)

    def reset(self):
        self.heat_map = self.initial_heat_map.copy()


def param_property(param):
    def getter(self):
        return self.params[param]

    def setter(self, value):
        self.params[param] = value

    return property(getter, setter)


class SimState:
    @classmethod
    def add_default_attributes(cls, defaults_dict):
        for param in defaults_dict.keys():
            setattr(cls, param, param_property(param))

    def __init__(self):
        self.running = False
        self.render_changes = False
        self.regen_plot = False
        self.params = {}

    def print_info(self):
        average_temp = self.plate.heat_map.mean().round(2)
        thickness = self.thickness
        dt = self.dt
        material = self.material.capitalize()
        energy_info = ut.total_energy(
            self.plate.p,
            self.plate.c,
            self.plate.heat_map,
            self.plate.dr**2 * thickness,
        )
        thermal_energy = energy_info[0].round(2)
        energy_units = energy_info[1]
        print(
            f"""
Material: {material}
Side Length: {self.plate.side_length}m
Thickness: {thickness}m
Points: {self.plate.points}x{self.plate.points}
Time Step: {dt}s
Average Temperature: {average_temp}K
Total Thermal Energy: {thermal_energy}{energy_units}
        """
        )

    def reset_flags(self):
        self.running = False
        self.render_changes = False
        self.regen_plot = False

    def update_plate(self):
        material = self.material
        points = self.points
        side_length = self.side_length
        function = self.function
        dt = self.dt
        new_min = self.min_temp
        new_max = self.max_temp
        try:
            new_plate = gen_plate(points, side_length, function, new_min, new_max)
        except InputError:
            raise
        self.plate = new_plate
        try:
            self.plate.gen_material_properties(material)
        except InitializationError:
            raise
        except InputError:
            raise
        self.plate.gen_solver(dt)
        self.render_changes = True

    def update_material(self, new_material):
        self.material = new_material
        try:
            self.plate.gen_material_properties(new_material)
        except InitializationError:
            raise
        except InputError:
            raise
        self.plate.gen_solver(self.dt)
        self.render_changes = True

    def update_thickness(self, new_thickness):
        self.thickness = new_thickness

    def update_dt(self, new_dt):
        self.dt = new_dt
        self.plate.gen_solver(new_dt)
        self.render_changes = True

    def start(self):
        self.running = True
        self.render_changes = True

    def stop(self):
        self.running = False
        self.render_changes = True

    def restart(self):
        self.plate.reset()
        self.running = False
        self.render_changes = True

    def step(self):
        self.plate.update()
        self.render_changes = True


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        raise InputError(message)

    def exit(self, status=0, message=None):
        os._exit(2)

    def populate(self):
        subs = self.add_subparsers(dest="cmd")

        new_cmd = subs.add_parser("new", help="initialize a new plate")
        new_cmd.add_argument("-f", "--function", type=str, help="function with which to generate the initial heat distribution")
        new_cmd.add_argument("-p", "--points", type=int, help="number of points per side with which to approximate the plate")
        new_cmd.add_argument("-m", "--material", type=str, help="material of the plate")
        new_cmd.add_argument("-s", "--side", type=float, help="side length of the plate in meters")
        new_cmd.add_argument("-t", "--time", type=float, help="time step of the simulation in seconds")
        new_cmd.add_argument("-th", "--thickness", type=float, help="thickness of the plate in meters")
        new_cmd.add_argument("-d", "--defaults", action="store_true", help="use default parameters")

        update_cmd = subs.add_parser("update", help="modify certain parameters")
        update_cmd.add_argument("-m", "--material", type=str, help="modify the material of the plate")
        update_cmd.add_argument("-s", "--side", type=float, help="modify the side length")
        update_cmd.add_argument("-t", "--time", type=float, help="modify the time step")
        update_cmd.add_argument("-th", "--thickness", type=float, help="modify the thickness")

        materials_cmd = subs.add_parser("materials", help="print a list of usable materials")
        functions_cmd = subs.add_parser("functions", help="print a list of functions")
        defaults_cmd = subs.add_parser("defaults", help="print the default parameters")
        start_cmd = subs.add_parser("start", help="start the simulation")
        stop_cmd = subs.add_parser("stop", help="stop the simulation")
        restart_cmd = subs.add_parser("restart", help="restart the simulation")
        exit_cmd = subs.add_parser("exit", help="exit the simulation")
        clear_cmd = subs.add_parser("clear", help="clear the screen")
        info_cmd = subs.add_parser("info", help="print detailed information about the current simulation")
        help_cmd = subs.add_parser("help", help="print a help message")


def gen_plate(points, side_length, function, new_min, new_max):
    try:
        fn = getattr(gen, f"{function}_map")
    except AttributeError:
            raise ParameterError(f"unknown function name {function}.")
    initial_map = fn(points, new_min, new_max)
    new_plate = Plate(initial_map, points, side_length)
    return new_plate


def input_loop(state):
    parser = MyParser(exit_on_error=False)
    parser.populate()
    
    while True:
        cmd = input("> ")
        try:
            args = parser.parse_args(cmd.split()) 
        except Exception as e:
            print("[WARN]", e)
            continue
        match args.cmd:
            case "help":
                print_help_message()

            case "materials":
                try:
                    info = ut.generate_materials_list(MATERIALS_PATH)
                except InitializationError as e:
                    print("[FATAL]", e)
                    os._exit(1)
                print(info)

            case "defaults":
                try:
                    info = ut.generate_defaults_info(DEFAULTS_PATH)
                except InitializationError as e:
                    print("[FATAL]", e)
                    os._exit(1)
                print(info)

            case "functions":
                try:
                    info = ut.generate_functions_list(FUNCTIONS_PATH)
                except InitializationError as e:
                    print("[FATAL]", e)
                    os._exit(1)
                print(info)

            case "info":
                with lock:
                    if not begin_sim.is_set():
                        print("[WARN] Cannot print info before initializing a plate.")
                        continue
                    state.print_info()

            case "start":
                with lock:
                    if not begin_sim.is_set():
                        print("[WARN] Cannot start before initializing a plate.")
                        continue
                    state.start()

            case "stop":
                with lock:
                    if not begin_sim.is_set():
                        print("[WARN] Cannot stop before initializing a plate.")
                        continue
                    state.stop()

            case "restart":
                with lock:
                    if not begin_sim.is_set():
                        print("[WARN] Cannot restart before initializing a plate.")
                        continue
                    state.restart()

            case "exit":
                os._exit(1)

            case "clear":
                ut.clear()

            case "new":
                with lock:
                    state.reset_flags()
                    try:
                        if not begin_sim.is_set():
                            defaults = ut.get_default_params(DEFAULTS_PATH)
                            state.add_default_attributes(defaults)
                            state.params = defaults
                        if args.defaults:
                            defaults = ut.get_default_params(DEFAULTS_PATH)
                            state.params = defaults
                        else:
                            for key, val in vars(args).items():
                                if val and key in state.params:
                                    if key == "points" and val != state.points:
                                        state.regen_plot = True
                                    state.params[key] = val
                        state.update_plate()
                        begin_sim.set()
                    except InputError as e:
                        print("[WARN]", e)
                        continue
                    except InitializationError as e:
                        print("[FATAL]", e)
                        os._exit(1)
            case "update":
                with lock:
                    if not begin_sim.is_set():
                        print("[WARN] Cannot update parameters before initializing a plate.")
                        continue
                    if args.time:
                        state.update_dt(args.time)
                    if args.material:
                        try:
                            state.update_material(args.material)
                        except InputError as e:
                            print("[WARN]", e)
                            continue
                        except InitializationError as e:
                            print("[FATAL]", e)
                            os._exit(1)
                    if args.thickness:
                        state.update_thickness(args.thickness)


def generate_plot_info(state):
    average_temp = state.plate.heat_map.mean().round(2)
    dt = state.dt
    material = state.material.capitalize()
    if state.running:
        status = "Running"
    else:
        status = "Paused"
    return f"Δt: {dt}s\nMaterial: {material}\nAverage Temp: {average_temp}K\nStatus: {status}"


def print_help_message():
    print(
        """
COMMANDS 
    • defaults
        Prints a list of the default parameters.
    • materials
        Prints a list of usable materials in descending order of conductivity.
    • functions
        Prints a list of usable functions. 
    • new {options}
        If no plate has been initialized, this will generate a new plate to be simulated. If a plate has already been initialized, this will stop the current simulation and generate a new initial distribution.
        new -d — Generates a plate with default parameters.
        new -f {function} {options} — Function with which to generate the initial heat distribution.
        new -m {material} {options} — Material of the plate.
        new -s {side length} {options} — Physical size of the grid in meters.
        new -p {points} {options} — Number of points per side with which the plate is approximated.
        new -t {time step} {options} — Time step with which to simulate the plate in seconds.
        new -th {thickness} {options} — Thickness of the plate in meters.
        If an option is not provided, its parameter will be copied from the previous plate (i.e. changes to parameters are persistent). If no plate has been initialized, the default parameters will be used. 
    • update {options}
        If a plate has been initialized, this will update the specified parameters. This command can be run at any time so long as a plate has been initialized. 
        update -m {material} {options} — Modifies the material of the plate.
        update -t {time step} {options} — Modifies the time step.
        update -th {thickness} {options} — Modifies the thickness of the plate.
    • start
        Starts the simulation.
    • stop
        Starts the simulation.
    • restart
        Restarts the simulation, restoring the plate to its initial state.
    • exit
        Exits the program.
    • info
        Prints detailed information about the current simulation.
    • help
        Prints this message.
          """
    )


def generate_plot(state):
    plt.style.use("dark_background")
    fig, axis = plt.subplots()
    fig.subplots_adjust(right=0.75)
    info = fig.text(
        0.78, 0.5, generate_plot_info(state), va="center", ha="left", family="monospace"
    )
    pcm = axis.pcolormesh(
        state.plate.heat_map, cmap=plt.cm.jet, vmin=state.min_temp, vmax=state.max_temp
    )
    bar = plt.colorbar(pcm, ax=axis)
    state.fig = fig
    state.axis = axis
    state.info = info
    state.pcm = pcm
    state.bar = bar


begin_sim = threading.Event()
lock = threading.Lock()

sim = SimState()

print('For a list of possible commands, use "help".')
thread = threading.Thread(target=input_loop, args=(sim,), daemon=True)
thread.start()

begin_sim.wait()

generate_plot(sim)

while True:
    with lock:
        if sim.regen_plot:
            plt.close()
            generate_plot(sim)
            sim.regen_plot = False
        if sim.running:
            sim.step()
        if sim.render_changes:
            sim.pcm.set_array(sim.plate.heat_map)
            sim.info.set_text(generate_plot_info(sim))
            sim.render_changes = False
    plt.pause(0.01)
plt.show()
