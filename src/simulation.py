import matplotlib.pyplot as plt
import numpy as np
import random, json, sys, threading, os
from backwards_euler import insert_matrix, gen_coeff_matrix, gen_known_vector, next_temps
import scipy.sparse as spr
import scipy.sparse.linalg as spl
import initial_gen as gen
from exceptions import *
from time import sleep
from pathlib import Path

base_dir = Path(__file__).parent
config_dir = base_dir.parent / "configs"
DEFAULTS_PATH = config_dir / "defaults.json"
MATERIALS_PATH = config_dir / "materials.json"

class Plate:
    def __init__(self, initial_heat_map, points, side_length):
        self.heat_map = initial_heat_map.copy()
        self.initial_heat_map = initial_heat_map.copy()
        self.points = points
        self.dr = side_length / (self.points - 1)

    def gen_solver(self, dt):
        self.coeff = self.diffusivity * dt / (self.dr ** 2) 
        coeff_matrix = gen_coeff_matrix(self.points-2, 1 + 4*self.coeff, -self.coeff)
        coeff_matrix = spr.csc_matrix(coeff_matrix)
        self.solve = spl.factorized(coeff_matrix)

    def gen_material_properties(self, material):
        try:
            with open(MATERIALS_PATH, 'r') as f:
                material_dict = json.load(f)
        except FileNotFoundError:
            raise MaterialsFileError("Could not find materials file.") 

        try:
            properties = material_dict[material]
        except KeyError:
            raise ParameterError(f"Could not get properties for \"{material}\".")

        try:
            k = properties["k"]
            p = properties["p"]
            c = properties["c"]
        except Exception as e:
            raise MaterialsFileError(f"Could not decode material properties: {e}.") 

        self.diffusivity = k / (p * c)

    def update(self):
        t = gen_known_vector(self.heat_map, self.coeff)
        new_temps_vec = self.solve(t) 
        new_temps_matrix = new_temps_vec.reshape(self.points-2, self.points-2)
        self.heat_map = insert_matrix(new_temps_matrix, self.heat_map, 1, 1)

    def reset(self):
        self.heat_map = self.initial_heat_map.copy()


class SimState:
    def __init__(self):
        self.running = False
        self.render_changes = False

    def reset_flags(self):
        self.running = False
        self.render_changes = False

    def add_plate(self, params):
        material = params["material"]
        points = params["points"]
        side_length = params["side_length"]
        function = params["function"]
        dt = params["dt"]
        try:
            new_plate = gen_plate(points, side_length, function)
        except InputError:
            raise
        self.plate = new_plate
        self.material = material
        self.dt = dt
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
        self.render_changes = True

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


def gen_plate(points, side_length, function):
    match function:
        case "poly":
            initial_map = gen.poly_map(points)
        case "piecewise_poly":
            initial_map = gen.piecewise_poly_map(points)
        case _:
            raise ParameterError("Unknown function name.")
    new_plate = Plate(initial_map, points, side_length)
    return new_plate

def new_state_args(cmds):
    try:
        with open(DEFAULTS_PATH, 'r') as f:
            try:
                params = json.load(f)
            except Exception as e:
                raise DefaultsFileError(f"Could not decode default parameters ({DEFAULTS_PATH}): {e}.") 
    except FileNotFoundError:
        raise DefaultsFileError("Could not find default parameters file.") 
    try:
        material = params["material"]
        points = params["points"]
        side_length = params["side_length"]
        function = params["function"]
        dt = params["dt"]
    except Exception as e:
        raise DefaultsFileError(f"Could not decode default parameters ({DEFAULTS_PATH}): {e}.") 
    cmds_string = "".join(cmds)
    separate_commands = [cmd for cmd in cmds_string.split('-') if cmd]
    for command in separate_commands:
        split_command = command.split()
        if len(split_command) > 2:
            raise ParameterError("Options cannot contain multiple words.")
        match split_command[0]:
            case 'f':
                params["function"] = split_command[1].lower()
            case 'p':
                try:
                    points = int(split_command[1])
                except ValueError:
                    raise IncompatibleTypeError("Invalid points parameter.")
                params["points"] = points
            case 'm':
                params["material"] = split_command[1].lower()
            case 's':
                try:
                    side_length = float(split_command[1])
                except ValueError:
                    raise IncompatibleTypeError("Invalid side length parameter.")
                params["side_length"] = side_length
            case 't':
                try:
                    dt = float(split_command[1])
                except ValueError:
                    raise IncompatibleTypeError("Invalid time step parameter.")
                params["dt"] = dt
            case _:
                raise ParameterError("Could not parse parameters.")
    return params


def input_loop(state):
    while True:
        cmd = input("> ")
        if cmd == "help":
            print_help_message()

        elif cmd == "start":
            with lock:
                if begin_sim.is_set():
                    state.start()
                else:
                    print("[WARN] Cannot start before initializing a plate.")

        elif cmd == "stop":
            with lock:
                if begin_sim.is_set():
                    state.stop()
                else:
                    print("[WARN] Cannot stop before initializing a plate.")

        elif cmd == "restart":
            with lock:
                if begin_sim.is_set():
                    state.restart()
                else:
                    print("[WARN] Cannot restart before initializing a plate.")

        elif cmd == "exit":
            os._exit(1)

        elif cmd == "clear":
            os.system("clear")

        elif cmd.split()[0] == "new":
            with lock:
                state.reset_flags()
                try:
                    sim_params = new_state_args(cmd[4:])
                    if begin_sim.is_set(): # If the simulation state already exists, it should inherit the time step and material
                        sim_params["dt"] = state.dt
                        sim_params["material"] = state.material
                    state.add_plate(sim_params)
                    begin_sim.set()
                except InputError as e:
                    print("[WARN]", e) 
                except InitializationError as e:
                    print("[FATAL]", e) 
                    os._exit(1)

        elif cmd.split()[0] == "time_step":
            with lock:
                new_dt = cmd.split()[1]
                try:
                    new_dt = float(new_dt)
                    if begin_sim.is_set():
                        state.update_dt(new_dt)
                    else:
                        print("[WARN] Cannot change the time step before initializing a plate.")
                except ValueError:
                    print(f"[WARN] Invalid time step \"{new_dt}\".") 

        elif cmd.split()[0] == "material":
            with lock:
                new_material = cmd.split()[1]
                if begin_sim.is_set():
                    try:
                        state.update_material(new_material)
                    except InitializationError as e:
                        print("[FATAL]", e) 
                        os._exit(1)
                    except InputError as e:
                        print("[WARN]", e)
                else:
                    print("[WARN] Cannot change the material before initializing a plate.")
        else:
            print(f"[WARN] Unknown command \"{cmd}\".")

       
def print_help_message():
    print("""
COMMANDS 
    • new {options}
        If no plate has been initialized, this will generate a new plate to be simulated. If a plate has already been initialized, this will stop the current simulation and generate a new initial distribution.
        new -f {function} - Function with which to generate the initial heat distribution. Defaults to a piecewise polynomial distribution.
        new -m {material} - Material of the plate. Defaults to aluminum.
        new -s {side length} - Physical size of the grid in meters. Defaults to 0.5.
        new -p {points} - Number of points with which the plate is approximated. Defaults to 100.
        new -t {time step} - Time step with which to simulate the plate. Defaults to 0.5.
        Options can be combined, e.g:
            new -f poly -s 0.1
    • time_step {time}
        Modifies the time step used for the simulation in seconds. Changing the time step is persistent across plates.
    • start
        Starts the simulation.
    • stop
        Starts the simulation.
    • restart
        Restarts the simulation, restoring the plate to its initial state.
    • exit
        Exits the program.
    • help
        Prints this message.
          """)

def generate_plot_info(state):
    average_temp = state.plate.heat_map.mean().round(2)
    if state.running:
        status = "Running"
    else:
        status = "Paused"
    return f"Δt: {state.dt}s\nMaterial: {state.material}\nAverage temp: {average_temp}K\nStatus: {status}"

begin_sim = threading.Event()
lock = threading.Lock()

sim = SimState()

print("For a list of possible commands and options, use \"help\".")
thread = threading.Thread(target=input_loop, args=(sim,), daemon=True)
thread.start()

begin_sim.wait()

plt.style.use('dark_background')
fig, axis = plt.subplots()
fig.subplots_adjust(right=0.75)

with lock:
    pcm = axis.pcolormesh(sim.plate.heat_map, cmap=plt.cm.jet, vmin=0, vmax=sim.plate.heat_map.max())
    info = fig.text(
        0.78, 0.5,
        generate_plot_info(sim),
        va="center",
        ha="left",
        family="monospace"
    )
plt.colorbar(pcm, ax=axis)

while True:
    with lock:
        if sim.running:
            sim.step()
        if sim.render_changes:
            pcm.set_array(sim.plate.heat_map)
            info.set_text(generate_plot_info(sim))
            sim.render_changes = False
    plt.pause(0.01)
plt.show()
