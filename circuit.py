import argparse
import matplotlib.pyplot as plt
import networkx as nx
import random
import yaml
import numpy as np

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Union, Tuple
from pyDOE2 import ccdesign, pbdesign

SECRET_VALUE = "s"

class InstructionType(Enum):
    INPUT = 1
    COMPUTATION = 2
    OUTPUT = 3


class InfiniteIO(list):
    def __len__(self):
        return 1


InfiniteIO = InfiniteIO()


class InstructionInputs(Dict):
    def __init__(self, inputs: List[str]):
        super(InstructionInputs, self).__init__()
        self.original = defaultdict(int)
        for i in inputs:
            self.original[i] += 1
        self.reset()

    def reset(self):
        for k, v in self.original.items():
            self[k] = v

class Instruction:
    def __init__(self, d: Dict[str, Union[str, int, float]]):
        self.name = str(d["name"])
        self.inputs = InstructionInputs(d["inputs"])
        self.outputs = d["outputs"]
        self.width = 0
        if "width" in d:
            self.width = int(d["width"])
        self.type = InstructionType.COMPUTATION
        if len(self.inputs) == 0:
            self.type = InstructionType.INPUT
        elif len(self.outputs) == 0:
            self.type = InstructionType.OUTPUT

    def __str__(self):
        return self.name


class SecretInputInstruction(Instruction):
    def __init__(self, party: int):
        self.party = party
        d = {
            "name": "INPUT%s" % (self.party),
            "inputs": [],
            "outputs": [SECRET_VALUE],
            "width": 0,
        }
        super(SecretInputInstruction, self).__init__(d)


class OutputInstruction(Instruction):
    def __init__(self, input_type: str):
        self.type_to_output = input_type
        d = {
            "name": "OUTPUT%s" % (self.type_to_output),
            "inputs": [self.type_to_output],
            "outputs": [],
            "width": 0,
        }
        super(OutputInstruction, self).__init__(d)


class Circuit:
    class Node:
        id = defaultdict(int)

        @staticmethod
        def reset():
            Circuit.Node.id = defaultdict(int)

        def __init__(self, key: str):
            self.width = 0
            self.key = key
            self.id = Circuit.Node.id[self.key]
            Circuit.Node.id[self.key] += 1

        def __str__(self):
            return "{}_{}".format(self.key, self.id)

    class RegisterNode(Node):
        def __init__(self, instruction: Instruction, reg_type: str):
            super().__init__(str(reg_type))
            self.reg_type = reg_type

    class InstructionNode(Node):
        def __init__(self, instruction: Instruction):
            super().__init__(str(instruction))
            self.instruction = instruction
            self.outputs = []
            for o_type in self.instruction.outputs:
                self.outputs.append(Circuit.RegisterNode(instruction, o_type))


    class InputNodes(Dict):
        def total_nodes(self):
            return sum((len(n) for n in self.values()))

    @staticmethod
    def linear_gate(num_gates, input_inst, gate_inst, output_inst) -> nx.DiGraph:
        Circuit.Node.reset()
        graph = nx.DiGraph()
        input1 = Circuit.Node(input_inst[0])
        input2 = Circuit.Node(input_inst[1])
        prev_gate = Circuit.Node(gate_inst)
        graph.add_nodes_from((input1, input2, prev_gate))
        graph.add_edge(input1, prev_gate)
        graph.add_edge(input2, prev_gate)
        for i in range(1, num_gates):
            gate = Circuit.Node(gate_inst)
            graph.add_node(gate)
            graph.add_edge(prev_gate, gate)
            graph.add_edge(input1, gate)
            prev_gate = gate
        output = Circuit.Node(output_inst)
        graph.add_node(output)
        graph.add_edge(prev_gate, output)
        return graph

    @staticmethod
    def new_interleave_gates(inputs: Dict[Instruction, int], gates: Dict[Instruction, int],
                             output: Dict[OutputInstruction, int], width: int, output_inst: Dict[str, OutputInstruction]) -> nx.DiGraph:
        Circuit.Node.reset()
        graph = nx.MultiDiGraph()
        input_nodes = Circuit._secret_inputs(graph, inputs)
        shuffled_input_nodes = {}
        unused_register_nodes = defaultdict(list)
        all_register_nodes = defaultdict(list)
        for key, nodes in input_nodes.items():
            shuffled_input_nodes[key] = nodes.copy()
            random.shuffle(shuffled_input_nodes[key])
            all_register_nodes[SECRET_VALUE].extend(nodes)
        gate_nodes = []
        curr_gate = None
        curr_width = 0
        input_i = 0
        for gate_inst in sorted(gates.keys(), key=lambda i: len(i.inputs)):
            cnt = gates[gate_inst]
            for _ in range(cnt):
                gate_inst.inputs.reset()
                gate_node = Circuit.InstructionNode(gate_inst)
                graph.add_node(gate_node)
                # Grow the depth of the graph
                if curr_width < width and gate_inst.width > 0:
                    gate_inst.inputs[curr_gate.reg_type] -= 1
                    graph.add_edge(curr_gate, gate_node)
                    for reg_type, reg_cnt in gate_inst.inputs.items():
                        for _ in range(reg_cnt):
                            if reg_type == curr_gate.reg_type:
                                in_node = curr_gate
                            elif len(unused_register_nodes[reg_type]) > 0:
                                in_node = unused_register_nodes[reg_type].pop(0)
                            else:
                                in_node = random.choice(all_register_nodes[reg_type])
                            graph.add_edge(in_node, gate_node)
                    curr_width += gate_inst.width
                    gate_node.width = curr_width
                    curr_gate = None
                    for out in gate_node.outputs:
                        graph.add_edge(gate_node, out)
                        out.width = gate_node.width
                        if curr_width < width:
                            if out.reg_type == SECRET_VALUE:
                                curr_gate = out
                            else:
                                all_register_nodes[out.reg_type].append(out)
                    if curr_width < width and curr_gate is None:
                        for out in gate_node.outputs:
                            for g in gates:
                                if out.reg_type in g.inputs and SECRET_VALUE in g.outputs:
                                    g.inputs.reset()
                                    next_node = Circuit.InstructionNode(g)
                                    graph.add_node(next_node)
                                    graph.add_edge(out, next_node)
                                    g.inputs[out.reg_type] -= 1
                                    for reg_type, reg_cnt in g.inputs.items():
                                        for _ in range(reg_cnt):
                                            if len(unused_register_nodes[reg_type]) > 0:
                                                in_node = unused_register_nodes[reg_type].pop(0)
                                            else:
                                                in_node = random.choice(all_register_nodes[reg_type])
                                            graph.add_edge(in_node, next_node)
                                    curr_width += g.width
                                    next_node.width = curr_width
                                    for out in next_node.outputs:
                                        graph.add_edge(next_node, out)
                                        out.width = curr_width
                                        if out.reg_type == SECRET_VALUE:
                                            curr_gate = out
                                        else:
                                            unused_register_nodes[out.reg_type].append(out)
                                            all_register_nodes[out.reg_type].append(out)
                                    gates[g] -= 1
                                    break
                    continue
                # Make sure inputs are mixed
                for reg_type, reg_cnt in gate_inst.inputs.items():
                    for i in range(reg_cnt):
                        if reg_type == SECRET_VALUE and reg_cnt > 1 and len(shuffled_input_nodes[input_i]) > 0:
                            in_node = shuffled_input_nodes[input_i].pop()
                            input_i = (input_i + 1) % len(input_nodes)
                        elif len(unused_register_nodes[reg_type]) > 0:
                            in_node = unused_register_nodes[reg_type].pop(0)
                        else:
                            in_node = random.choice(all_register_nodes[reg_type])
                        graph.add_edge(in_node, gate_node)
                        gate_node.width = max(gate_node.width, in_node.width)
                for out in gate_node.outputs:
                    graph.add_edge(gate_node, out)
                    out.width = gate_node.width
                    if out.width < width:
                        unused_register_nodes[out.reg_type].append(out)
                        all_register_nodes[out.reg_type].append(out)
                    if curr_gate is None and out.reg_type == SECRET_VALUE and len(gate_node.outputs) == 1:
                        curr_gate = out
                gates[gate_inst] -= 1
        needs_output = defaultdict(list)
        num_outputs = defaultdict(int)
        for reg_type, reg_nodes in all_register_nodes.items():
            for reg_node in reg_nodes:
                if graph.out_degree(reg_node) == 0:
                    needs_output[reg_type].append(reg_node)
        for out_inst, cnt in output.items():
            for _ in range(cnt):
                reg_type = out_inst.type_to_output
                out_node = Circuit.InstructionNode(out_inst)
                graph.add_node(out_node)
                if len(needs_output[reg_type]) > 0:
                    in_node = needs_output[reg_type].pop()
                else:
                    in_node = random.choice(all_register_nodes[reg_type])
                graph.add_edge(in_node, out_node)
                num_outputs[reg_type] += 1
        for reg_type, reg_nodes in needs_output.items():
            for reg_node in reg_nodes:
                out_node = Circuit.InstructionNode(output_inst[reg_type])
                graph.add_node(out_node)
                graph.add_edge(reg_node, out_node)
                num_outputs[reg_type] += 1
        nx.readwrite.write_adjlist(graph, "test.circuit")
        return graph, num_outputs

    @staticmethod
    def interleave_gates(inputs: Dict[Instruction, int], gates: Dict[Instruction, int], output: Dict[Instruction, int], width: int) -> nx.DiGraph:
        Circuit.Node.reset()
        graph = nx.DiGraph()
        input_nodes = Circuit._inputs(graph, inputs)
        shuffled_input_nodes = Circuit.InputNodes()
        for i, nodes in zip(range(len(input_nodes)), input_nodes):
            shuffled_input_nodes.append(nodes.copy())
            random.shuffle(shuffled_input_nodes[i])
        gate_nodes = []
        input_i = 0
        curr_gate = None
        curr_width = 0
        for gate_inst, cnt in gates.items():
            for i in range(cnt):
                gate_node = Circuit.Node(gate_inst)
                graph.add_node(gate_node)
                n_inputs = gate_inst.inputs
                if curr_width < width and curr_gate is not None:
                    n_inputs = gate_inst.inputs - 1
                    graph.add_edge(curr_gate, gate_node)
                    curr_width += gate_inst.width
                for _ in range(n_inputs):
                    if shuffled_input_nodes.total_nodes() > 0:
                        if len(shuffled_input_nodes[input_i]) > 0:
                            in_node = shuffled_input_nodes[input_i].pop()
                        else:
                            in_node = random.choice(input_nodes[input_i])
                    else:
                        in_node = random.choice(input_nodes[input_i])
                    graph.add_edge(in_node, gate_node)
                    input_i = (input_i + 1) % len(inputs)
                gate_nodes.append(gate_node)
                curr_gate = gate_node
        need_output = []
        num_outputs = 0
        for gate_node in gate_nodes:
            if graph.out_degree(gate_node) < gate_node.instruction.outputs:
                need_output.append(gate_node)
        for output_inst, cnt in output.items():
            for _ in range(cnt):
                out_node = Circuit.Node(output_inst)
                graph.add_node(out_node)
                if len(need_output) > 0:
                    in_node = need_output.pop()
                else:
                    in_node = random.choice(gate_nodes)
                graph.add_edge(in_node, out_node)
                num_outputs += 1
        output_inst = list(output.keys())[0]
        for gate in need_output:
            out_node = Circuit.Node(output_inst)
            graph.add_node(out_node)
            graph.add_edge(gate, out_node)
            num_outputs += 1

        return graph, num_outputs

    @staticmethod
    def _secret_inputs(graph: nx.DiGraph, inputs: Dict[SecretInputInstruction, int]) -> Dict[Tuple[str, int], List[RegisterNode]]:
        input_nodes = defaultdict(list)
        for input_inst, cnt in inputs.items():
            for _ in range(cnt):
                input_node = Circuit.InstructionNode(input_inst)
                graph.add_node(input_node)
                for n in input_node.outputs:
                    graph.add_node(n)
                    graph.add_edge(input_node, n)
                    input_nodes[input_inst.party].append(n)
        return input_nodes

    @staticmethod
    def inputs(inputs: Dict[Instruction, int]) -> nx.DiGraph:
        Circuit.Node.reset()
        graph = nx.DiGraph()
        Circuit._inputs(graph, inputs)
        return graph

    @staticmethod
    def n_gate(inst: Instruction, inputs: Dict[Instruction, int], output: Instruction) -> nx.DiGraph:
        Circuit.Node.reset()
        graph = nx.DiGraph()
        input_nodes = Circuit._inputs(graph, inputs)
        gate_nodes = []
        for idx in range(len(input_nodes)):
            for i in input_nodes[idx]:
                gate_node = Circuit.Node(inst)
                graph.add_node(gate_node)
                graph.add_edge(i, gate_node)
                gate_nodes.append(gate_node)
        for g in gate_nodes:
            output_node = Circuit.Node(output)
            graph.add_node(output_node)
            graph.add_edge(g, output_node)
        return graph


class MPC:
    def __init__(self, spec_path: str):
        self._parse_spec(spec_path)

    def _parse_spec(self, spec_path: str):
        with open(spec_path, "r") as f:
            data = yaml.load(f, yaml.FullLoader)
        self.types = data["types"]
        self.all_inst = {}
        self.gate_inst = []
        self.input_inst = []
        self.output_inst = {}
        print(data)
        for i in range(data["parties"]):
            input_inst = SecretInputInstruction(i)
            self.input_inst.append(input_inst)
            self.all_inst[input_inst.name] = input_inst
        for t in self.types:
            output_inst = OutputInstruction(t)
            self.output_inst[t] = output_inst
        for i in data["instructions"]:
            i = Instruction(i)
            self.all_inst[i.name] = i
            self.gate_inst.append(i)

    def circuits(self):
        inputs = {}
        gates = {}
        for inst in self.input_inst:
            inputs[inst] = 0
        for inst in self.gate_inst:
            gates[inst] = 0

        for i in range(2, 16):
            n_gates = 2**i
            #n_gates = i
            for gate_inst in self.gate_inst:
                graph = Circuit.linear_gate(n_gates, self.input_inst, gate_inst, self.output_inst)
                #nx.readwrite.write_adjlist(graph, "{}_{}{}.circuit".format("linear", n_gates, gate_inst))
            for inst in self.input_inst:
                inputs[inst] = n_gates // len(self.input_inst)
            for inst in self.gate_inst:
                gates[inst] = n_gates
            graph = Circuit.inputs(inputs)
            #nx.readwrite.write_adjlist(graph, "input_{}.circuit".format(n_gates))
            graph = Circuit.interleave_gates(inputs, gates, {self.output_inst: n_gates}, n_gates)
            nx.readwrite.write_adjlist(graph, "interleave_{}.circuit".format(n_gates))
            graph = Circuit.n_gate(self.gate_inst[0], inputs, self.output_inst)
            #nx.readwrite.write_adjlist(graph, "gates_{}.circuit".format(n_gates))

        # colors = []
        # for n in G.nodes:
        #     if n.instruction == self.input_inst[0]:
        #         colors.append("green")
        #     else:
        #         colors.append("red")
        # nx.draw_kamada_kawai(G, node_color=colors)


def optimal_mixing(mpc):
    inputs = {}
    for i in [1, 2, 5, 10, 25, 50, 100, 200, 300, 500, 800]:
        for inst in mpc.input_inst:
            inputs[inst] = i
        for inst in mpc.gate_inst:
            graph, _ = Circuit.interleave_gates(inputs, {inst: i}, {mpc.output_inst: i}, width=0)
            print(i, inst)
            nx.readwrite.write_adjlist(graph, "{}{}-1w.circuit".format(i, inst.name))


def cheap_smc(mpc):
    for inst in mpc.gate_inst:
        graph = Circuit.linear_gate(1000, mpc.input_inst, inst, mpc.output_inst)
        nx.readwrite.write_adjlist(graph, "1000-{}.circuit".format(inst.name))


def random_circuits(mpc):
    random.seed(0)
    MAX = 2**16
    MAX_WIDTH = 2**10
    for i in range(100):
        inputs = {}
        gates = {}
        gate_str = []
        n_inputs = random.randint(1, MAX)
        gate_counts = [random.randint(1, MAX), random.randint(1, MAX)]
        n_outputs = random.randint(1, sum(gate_counts))
        width = random.randint(1, MAX_WIDTH)
        if width > gate_counts[-1]:
            width = gate_counts[-1]
        for inst in mpc.input_inst:
            inputs[inst] = n_inputs // len(mpc.input_inst)
        output = {mpc.output_inst: n_outputs}
        for inst, n_gates in zip(mpc.gate_inst, gate_counts):
            gates[inst] = n_gates
            gate_str.append("{}{}".format(n_gates, inst.name))
        graph, adjusted_outputs = Circuit.interleave_gates(inputs, gates, output, width)
        if adjusted_outputs != n_outputs:
            print("adjusted:", adjusted_outputs, "from", n_outputs)
        nx.readwrite.write_adjlist(graph, "rand{}i-{}o-{}-{}w.circuit".format(
            n_inputs, adjusted_outputs, "-".join(gate_str), width))
    return

def pbd_no_width(mpc):
    inputs = {}
    gates = {}
    pbconfigs = pbdesign(4)
    for config in pbconfigs:
        scaled = []
        for value in config:
            scaled.append(int(max(2, (max_val/ 2) + (value * (max_val / 2)))))
        print(scaled)
        n_inputs = scaled[0]
        n_outputs = scaled[1]
        gate_counts = scaled[2:]
        gate_str = []
        for inst in mpc.input_inst:
            inputs[inst] = n_inputs // len(mpc.input_inst)
        output = {mpc.output_inst: n_outputs}
        for inst, n_gates in zip(mpc.gate_inst, gate_counts):
            gates[inst] = n_gates
            gate_str.append("{}{}".format(n_gates, inst.name))
        graph, adjusted = Circuit.interleave_gates(inputs, gates, output, width=0)
        if adjusted != n_outputs:
            print("adjusted:", adjusted, "from", n_outputs)
        nx.readwrite.write_adjlist(graph, "{}i-{}o-{}-{}w.circuit".format(
            n_inputs, adjusted, "-".join(gate_str), 0))

def pbd(mpc, max_val=2**13):
    inputs = {}
    gates = {}
    #pbconfigs = pbdesign(len(mpc.gate_inst) + 1)

    circuits = set()
    pbconfigs = pbdesign(5)
    for i, config in enumerate(pbconfigs):
        config = config[:len(mpc.gate_inst) + 2]
        scaled = []
        for value in config:
            scaled.append(int(max(3, (max_val/ 2) + (value * (max_val / 2)))))
        print(scaled)
        n_inputs = scaled[0]
        #n_outputs = scaled[1]
        #gate_counts = scaled[2:-1]
        gate_counts = scaled[1:-1]
        gate_str = []
        width = scaled[-1]
        if width > scaled[-2]:
            width = scaled[-2]
        for inst in mpc.input_inst:
            inputs[inst] = n_inputs // len(mpc.input_inst)
        #output = {mpc.output_inst: n_outputs}
        output = {}
        for inst, n_gates in zip(mpc.gate_inst, gate_counts):
            gates[inst] = n_gates
            gate_str.append("{}{}".format(n_gates, inst.name))
        max_width = sum([n_gates * inst.width for inst, n_gates in gates.items()])
        if max_width < width:
            print("adjusting gates to meet width: %d (curr max: %d)" % (width, max_width))
            difference = width - max_width
            inst_widths = sum([inst.width for inst in gates])
            for inst in gates:
                gates[inst] += int(inst.width * (difference / inst_widths))
        graph, adjusted = Circuit.new_interleave_gates(inputs, gates, output, width, mpc.output_inst)
        #if adjusted != n_outputs:
        #    print("adjusted:", adjusted, "from", n_outputs)

        circuit_desc = "{}i-{}-{}w".format(n_inputs, "-".join(gate_str), width)
        print(circuit_desc)
        if circuit_desc in circuits:
            continue
        circuits.add(circuit_desc)

        file_name = "out_circ/pbd_%d.circuit" % len(circuits)
        write_metadata(file_name, inputs, gates, output, width)
        with open(file_name, "ab") as f:
            nx.readwrite.write_adjlist(graph, f)
            #nx.readwrite.write_adjlist(graph, "{}i-{}o-{}-{}w.circuit".format(
            #    n_inputs, adjusted, "-".join(gate_str), width))


def write_metadata(file_name, inputs, gates, output, width):
    with open(file_name, "w") as f:
        for inp, cnt in inputs.items():
            print("# %s: %d" % (inp, cnt), file=f)
        for g, cnt in gates.items():
            print("# %s: %d" % (g, cnt), file=f)
        for out, cnt in output.items():
            print("# %s: %d" % (out, cnt), file=f)
        print("# width: %d" % width, file=f)



def ccd_no_width(mpc, max_val=2**13, width=0):
    dimensions = len(mpc.gate_inst) + 1
    print("dimensions=", dimensions)
    ccdconfigs = ccdesign(dimensions, center=(0, 1), face="circumscribed")
    inputs = {}
    gates = {}
    circuits = set()
    for config in ccdconfigs:
        scaled = []
        for value in config:
            scaled.append(int(max(2, (max_val/ 2) + (value * (max_val / 2)))))
        print(scaled)
        n_inputs = scaled[0]
        n_outputs = scaled[1]
        gate_counts = scaled[2:]
        gate_str = []
        for inst in mpc.input_inst:
            inputs[inst] = n_inputs // len(mpc.input_inst)
        #output = {mpc.output_inst: n_outputs}
        output = {}
        for inst, n_gates in zip(mpc.gate_inst, gate_counts):
            gates[inst] = n_gates
            gate_str.append("{}{}".format(n_gates, inst.name))
        graph, adjusted = Circuit.new_interleave_gates(inputs, gates, output, width, mpc.output_inst)
        #if adjusted != n_outputs:
        #    print("adjusted:", adjusted, "from", n_outputs)

        circuit_desc = "{}i-{}-{}w".format(n_inputs, "-".join(gate_str), width)
        print(circuit_desc)
        if circuit_desc in circuits:
            continue
        circuits.add(circuit_desc)

        file_name = "out_circ/ccd_%d.circuit" % len(circuits)
        #file_name = "out_circ/%s" % circuit_desc
        write_metadata(file_name, inputs, gates, output, width)
        with open(file_name, "ab") as f:
            nx.readwrite.write_adjlist(graph, f)


def ccd(mpc, max_val=2**13):
    inputs = {}
    gates = {}
    dimensions = len(mpc.gate_inst) + 2

    circuits = set()
    ccdconfigs = ccdesign(dimensions, center=(0, 1), face="circumscribed")
    for config in ccdconfigs:
        scaled = []
        for value in config:
            scaled.append(int(max(2, (max_val/ 2) + (value * (max_val / 2)))))
        print(scaled)
        n_inputs = scaled[0]
        n_outputs = scaled[1]
        gate_counts = scaled[2:-1]
        gate_str = []
        width = scaled[-1]
        if width > scaled[-2]:
            width = scaled[-2]
        for inst in mpc.input_inst:
            inputs[inst] = n_inputs // len(mpc.input_inst)
        #output = {mpc.output_inst: n_outputs}
        output = {}
        for inst, n_gates in zip(mpc.gate_inst, gate_counts):
            gates[inst] = n_gates
            gate_str.append("{}{}".format(n_gates, inst.name))
        graph, adjusted = Circuit.new_interleave_gates(inputs, gates, output, width, mpc.output_inst)
        #if adjusted != n_outputs:
        #    print("adjusted:", adjusted, "from", n_outputs)
        circuit_desc = "{}i-{}-{}w".format(n_inputs, "-".join(gate_str), width)
        print(circuit_desc)
        if circuit_desc in circuits:
            continue
        circuits.add(circuit_desc)

        file_name = "out_circ/ccd_%d.circuit" % len(circuits)
        #file_name = "out_circ/%s" % circuit_desc
        write_metadata(file_name, inputs, gates, output, width)
        with open(file_name, "ab") as f:
            nx.readwrite.write_adjlist(graph, f)


def interleave(mpc):
    inputs = {}
    gates = {}
    for i in range(1, 17):
        n_gates = 2**i
        width = min(n_gates, 2**10)
        for inst in mpc.input_inst:
            inputs[inst] = n_gates // len(mpc.input_inst)
        output = {mpc.output_inst: n_gates}
        gate_str = []
        for inst in mpc.gate_inst:
            gates[inst] = n_gates
            gate_str.append("{}{}".format(n_gates, inst.name))
        graph, adjusted = Circuit.interleave_gates(inputs, gates, output, width)
        if adjusted != n_gates:
            print("adjusted:", adjusted, "from", n_gates)
        nx.readwrite.write_adjlist(graph, "{}i-{}o-{}-{}w.circuit".format(
            n_gates, adjusted, "-".join(gate_str), width))


def main():
    parser = argparse.ArgumentParser(description='Circuit generator for CostCO.')
    parser.add_argument('spec_file', type=str, help='spec file path')
    parser.add_argument('experiment_type', type=str, help='experiment type (ccd, pbd)')
    parser.add_argument('-w', '--width', default=0, help='fixed width (circuit depth) [default=0]')
    parser.add_argument('-g', '--max_gates', default=2**13, help='max gates [default=2**13]')
    args = parser.parse_args()

    protocol = MPC(args.spec_file)
    if args.experiment_type == "pbd":
        pbd(protocol, args.max_gates)
    elif args.experiment_type == "ccd":
        if vary_width:
            ccd(protocol, args.max_gates)
        else:
            ccd_no_width(protocol, args.max_gates, args.width)


def _main():
    spdz = MPC("spdz.yaml")
    spdz = MPC("aby-yao.yaml")
    pbd(spdz)
    return
    Circuit.new_interleave_gates({i: 4 for i in spdz.input_inst}, {i: 4 for i in spdz.gate_inst}, {i: 4 for i in spdz.output_inst.values()}, 1, spdz.output_inst)
    return
    aby = MPC("a2y.yaml")
    ccd_no_width(aby)
    return
    pbd_no_width(aby)
    return
    aby = MPC("aby-yao.yaml")
    interleave(aby)
    return
    random.seed(0)
    for i in range(100):
        print(random.randint(1, 2**16))
    return
    #aby = MPC("aby.yaml")
    #optimal_mixing(aby)
    #return
    aby = MPC("aby-yao.yaml")
    aby = MPC("aby.yaml")

    ccd(aby)
    #pbd(aby)
    #ccd_no_width(aby)
    return
    #MPC("aby-yao.yaml").circuits()
    #print(ccdesign(4, center=(0, 1), face="inscribed"))
    inputs = {}
    gates = {}
    max_val = 2**13

    print(pbdesign(5))
    pbconfigs = pbdesign(5)
    ccdconfigs = ccdesign(3, center=(0, 1), face="circumscribed")
    ccdconfigs_reuse = []
    for c in ccdconfigs:
        ccdconfigs_reuse.append(c)
        for p in pbconfigs:
            if tuple(c) == tuple(p):
                ccdconfigs_reuse.pop()
    print(ccdconfigs)
    print(np.array(ccdconfigs_reuse))
    print(ccdesign(3, center=(0, 1), face="circumscribed"))
    print(np.array(ccdconfigs_reuse))
    for config in ccdconfigs_reuse:
        scaled = []
        for value in config:
            scaled.append(int(max(2, (max_val/ 2) + (value * (max_val / 2)))))
        print(scaled)
        n_inputs = scaled[0]
        #n_outputs = scaled[1]
        n_outputs = 1
        #gate_counts = scaled[2:-1]
        gate_counts = scaled[1:]
        gate_str = []
        width = scaled[-1]
        width = 2
        if width > scaled[-2]:
            width = scaled[-2]
        for inst in aby.input_inst:
            inputs[inst] = n_inputs // len(aby.input_inst)
        output = {aby.output_inst: n_outputs}
        for inst, n_gates in zip(aby.gate_inst, gate_counts):
            gates[inst] = n_gates
            gate_str.append("{}{}".format(n_gates, inst.name))
        graph, adjusted_outputs = Circuit.interleave_gates(inputs, gates, output, width)
        if adjusted_outputs != n_outputs:
            print("adjusted:", adjusted_outputs, "from", n_outputs)
        nx.readwrite.write_adjlist(graph, "{}i-{}o-{}-{}w.circuit".format(
            n_inputs, adjusted_outputs, "-".join(gate_str), width))
    return


    #MPC("a2y.yaml").circuits()
    #MPC("agmpc.yaml").circuits()


if __name__ == '__main__':
    main()
