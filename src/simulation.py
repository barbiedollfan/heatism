import matplotlib.pyplot as plt
import numpy as np
import random, json, sys, threading, os
from backwards_euler import *
from initial_gen import *
from exceptions import *
from time import sleep

class Plate:
    def __init__(self, material, initial_heat_map, points, side_length):
        try:
            with open('materials.json', 'r') as f:
                material_dict = json.load(f)
        except FileNotFoundError:
            raise MaterialsFileError("Could not find materials file.") 

        try:
            k = material_dict[material]["k"]
            p = material_dict[material]["p"]
            c = material_dict[material]["c"]
        except Exception:
            raise MaterialsFileError("Could not decode material properties.") 

        self.heat_map = initial_heat_map.copy()
        self.initial_heat_map = initial_heat_map.copy()
        self.points = points
        self.dr = side_length / (self.points - 1)
        self.diffusivity = k / (p * c)
        self.dt = 0.5
        self.coeff = self.diffusivity * self.dt / (self.dr ** 2) 
        coeff_matrix = gen_coeff_matrix(self.points-2, 1 + 4*self.coeff, -self.coeff)
        coeff_matrix = spr.csc_matrix(coeff_matrix)
        self.solve = spl.factorized(coeff_matrix)

    def step_simulation(self):
        self.heat_map = next_temps(self.solve, self.heat_map, self.coeff)

    def restart_simulation(self):
        self.heat_map = self.initial_heat_map.copy()


class SimState:
    def __init__(self):
        self.running = False
        self.render_changes = False

    def update_plate(self, plate):
        self.plate = plate
        self.render_changes = True
        self.running = False

    def update_dt(self, new_dt):
        if hasattr(self, 'plate'):
            plate = self.plate
            plate.dt = new_dt
            plate.coeff = plate.diffusivity * plate.dt / (plate.dr ** 2) 
            coeff_matrix = gen_coeff_matrix(plate.points-2, 1 + 4*plate.coeff, - plate.coeff)
            coeff_matrix = spr.csc_matrix(coeff_matrix)
            plate.solve = spl.factorized(coeff_matrix)
        else:
            raise UninitializedError("Cannot modify the time step before initializing the plate.") 

    def start(self):
        if hasattr(self, 'plate'):
            self.running = True
        else:
            raise UninitializedError("Cannot start before initializing the plate.") 

    def stop(self):
        if hasattr(self, 'plate'):
            self.running = False
        else:
            raise UninitializedError("Cannot stop before initializing the plate.") 

    def restart(self):
        if hasattr(self, 'plate'):
            self.plate.restart_simulation()
            self.render_changes = True
            self.running = False
        else:
            raise UninitializedError("Cannot restart before initializing the plate.") 

    def step(self):
        self.plate.step_simulation()
        self.render_changes = True


def parse_args(cmds):
    params = {"material": "aluminum",
              "points": 100,
              "side_length": 0.5,
              "function": "piecewise_poly"}
    cmds_string = "".join(cmds)
    separate_commands = [cmd for cmd in cmds_string.split('-') if cmd]
    for command in separate_commands:
        split_command = command.split()
        if len(split_command) > 2:
            raise ParameterError("Options cannot contain multiple words.")
        match split_command[0]:
            case 'f':
                params["function"] = split_command[1]
            case 'p':
                try:
                    points = int(split_command[1])
                except ValueError:
                    raise IncompatibleTypeError("Invalid points parameter.")
                params["points"] = points
            case 'm':
                params["material"] = split_command[1]
            case 's':
                try:
                    side_length = float(split_command[1])
                except ValueError:
                    raise IncompatibleTypeError("Invalid side length parameter.")
                params["side_length"] = side_length
            case _:
                raise ParameterError("Could not parse parameters.")
    return params


def generate_plate(args):
    function = args["function"] 
    points = args["points"] 
    material = args["material"] 
    side_length = args["side_length"] 
    match function:
        case "poly":
            initial_map = poly_map(points)
        case "piecewise_poly":
            initial_map = piecewise_poly_map(points)
        case _:
            raise ParameterError("Unknown function name.")
    try:
        plate = Plate(material.lower(), initial_map, points, side_length)
    except InitializationError:
        raise
    return plate


def input_loop(state):
    while True:
        cmd = input("> ")
        if cmd == "help":
            print_help_message()
        elif cmd == "start":
            try:
                state.start()
            except UninitializedError as e:
                print(f"[WARN]", e) 
        elif cmd == "restart":
            try:
                state.restart()
            except UninitializedError as e:
                print(f"[WARN]", e) 
        elif cmd == "stop":
            try:
                state.stop()
            except UninitializedError as e:
                print(f"[WARN]", e) 
        elif cmd == "exit":
            os._exit(1)
        elif cmd == "clear":
            os.system("clear")
        elif cmd.split()[0] == "new":
            try:
                plate_params = parse_args(cmd[4:])
                try:
                    plate = generate_plate(plate_params)
                    state.update_plate(plate)
                except InitializationError as e:
                    print("[FATAL]", e) 
                    os._exit(1)
                except InputError as e:
                    print("[WARN]", e)
            except InputError as e:
                print("[WARN]", e) 
        elif cmd.split()[0] == "time_step":
            new_dt = cmd.split()[1]
            try:
                new_dt = float(new_dt)
                try:
                    state.update_dt(new_dt)
                except InputError as e:
                    print(f"[WARN]", e) 
            except ValueError:
                print(f"[WARN] Invalid time step \"{new_dt}\".") 
        else:
            print(f"[WARN] Unknown command \"{cmd}\".")

       
def print_help_message():
    print("""
COMMANDS 
    • new {options}
        If no plate has been initialized, this will generate a new plate to be simulated. If a plate has already been initialized, this will stop the current simulation and generate a new initial distribution.
        new -f {function} - Function with which to generate the initial heat distribution. Defaults to a piecewise polynomial distribution.
        new -m {material} - Material of the plate. Defaults to aluminum.
        new -s {side_length} - Physical size of the grid in meters. Defaults to 0.5.
        new -p {points} - Number of points with which the plate is approximated. Defaults to 100.
        Options can be combined, e.g:
            new -f poly -s 0.1
    • time_step {time}
        Modifies the time step used for the simulation in seconds. Defaults to 0.5.
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


sim = SimState()

print("For a list of possible commands and options, use \"help\".")
t = threading.Thread(target=input_loop, args=(sim,), daemon=True)
t.start()

while not hasattr(sim, 'plate'):
    sleep(0.1)

plt.style.use('dark_background')
fig, axis = plt.subplots()
pcm = axis.pcolormesh(sim.plate.heat_map, cmap=plt.cm.jet, vmin=0, vmax=sim.plate.heat_map.max())
plt.colorbar(pcm, ax=axis)

while True:
    if sim.running:
        sim.step()
    if sim.render_changes:
        pcm.set_array(sim.plate.heat_map)
        sim.render_changes = False
    plt.pause(0.01)
plt.show()
