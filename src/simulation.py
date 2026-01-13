import matplotlib.pyplot as plt
import numpy as np
import random, json, sys, threading, os
from backwards_euler import *
from initial_gen import *
from time import sleep

class SimulationError(Exception):
    pass

class InputError(SimulationError):
    pass

class ParameterError(InputError):
    pass

class IncompatibleTypeError(InputError):
    pass

class Plate:
    def __init__(self, material, initial_heat_map, points, side_length):
        with open('materials.json', 'r') as f:
            material_dict = json.load(f)
        k = material_dict["plate"][material]["k"]
        p = material_dict["plate"][material]["p"]
        c = material_dict["plate"][material]["c"]
        self.heat_map = initial_heat_map.copy()
        self.initial_heat_map = initial_heat_map.copy()
        self.points = points
        self.dr = side_length / (self.points - 1)
        self.diffusivity = k / (p * c)
        self.dt = 1
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
        self.plate = None
        self.running = False
        self.render_changes = False
    def update_plate(self, plate):
        self.plate = plate
        self.render_changes = True
        self.running = False
    def start(self):
        if not self.plate:
            print("[WARN] Cannot start before initializing the plate.") 
        else:
            self.running = True
    def stop(self):
        if not self.plate:
            print("[WARN] Cannot stop before initializing the plate.") 
        else:
            self.running = False
    def restart(self):
        if not self.plate:
            print("[WARN] Cannot restart before initializing the plate.") 
        else:
            self.plate.restart_simulation()
            self.render_changes = True
            self.running = False
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
    return Plate(material.lower(), initial_map, points, side_length)


def input_loop(state):
    while True:
        cmd = input("> ")
        if cmd == "help":
            print_help_message()
        elif cmd == "start":
            state.start()
        elif cmd == "restart":
            state.restart()
        elif cmd == "stop":
            state.stop()
        elif cmd == "exit":
            os._exit(1)
        elif cmd == "clear":
            os.system("clear")
        elif cmd.split()[0] == "new":
            try:
                plate_params = parse_args(cmd[4:])
            except InputError as e:
                print("[FATAL]", e) 
                os._exit(1)
            try:
                plate = generate_plate(plate_params)
            except SimulationError as e:
                print("[FATAL]", e) 
                os._exit(1)
            state.update_plate(plate)
        else:
            print(f"[WARN] Unknown command \"{cmd}\".")

       
def print_help_message():
    print("""
COMMANDS 
    • new {options}
        If no plate has been initialized, this will generate a new plate to be simulated. If a plate has already been initialized, this will stop the current simulation and generate a new initial distribution.
        new -f {function} - Function with which to generate the initial heat distribution. Defaults to a piecewise polynomial distribution.
        new -f {material} - Material of the plate. Defaults to aluminum.
        new -f {side_length} - Physical size of the grid in meters. Defaults to 0.5.
        new -f {points} - Number of points with which the plate is approximated. Defaults to 100.
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

while not sim.plate:
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
