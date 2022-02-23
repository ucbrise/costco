import ast
import astor
import logging

from collections import defaultdict, namedtuple
from typing import Dict, List, Union, IO

from .mpc_source import CFG, Assignment, arith, bool, yao, Conversion
from .types import cint, sint, sintarray, cbool, is_secret, Input, Role
from .protocols import Circuit, Arithmetic, Boolean, Yao, A2Y, A2B, B2A, B2Y, Y2B, Y2A
from .ssa import remove_version

PROTOCOLS = [bool, yao, arith]
prelude = """
#include <iostream>
#include <fstream>
#include <vector>

//Utility libs
#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>

//ABY Party class
#include "../../abycore/aby/abyparty.h"
#include "../../abycore/circuit/booleancircuits.h"
#include "../../abycore/circuit/arithmeticcircuits.h"
#include "../../abycore/sharing/sharing.h"
int32_t read_test_options(int32_t* argcp, char*** argvp, e_role* role,
    uint32_t* bitlen, uint32_t* secparam, uint32_t* nrounds, std::string* address, uint16_t* port, int32_t* test_op) {

  uint32_t int_role = 0, int_port = 0;

  parsing_ctx options[] =
      { { (void*) &int_role, T_NUM, "r", "Role: 0/1", true, false }, {
          (void*) bitlen, T_NUM, "b", "Bit-length, default 32", false,
          false }, { (void*) secparam, T_NUM, "s",
          "Symmetric Security Bits, default: 128", false, false }, {
          (void*) nrounds, T_NUM, "i",
          "Number of rounds", false, false }, {
          (void*) address, T_STR, "a",
          "IP-address, default: localhost", false, false }, {
          (void*) &int_port, T_NUM, "p", "Port, default: 7766", false,
          false }, { (void*) test_op, T_NUM, "t",
          "Single test (leave out for all operations), default: off",
          false, false } };

  if (!parse_options(argcp, argvp, options,
      sizeof(options) / sizeof(parsing_ctx))) {
    print_usage(*argvp[0], options, sizeof(options) / sizeof(parsing_ctx));
    std::cout << "Exiting" << std::endl;
    exit(0);
  }

  assert(int_role < 2);
  *role = (e_role) int_role;

  if (int_port != 0) {
    assert(int_port < 1 << (sizeof(uint16_t) * 8));
    *port = (uint16_t) int_port;
  }

  //delete options;

  return 1;
}

share* put_cons32_gate(Circuit* c, uint32_t val) {
  uint32_t x = val;
  return c->PutCONSGate(x, (uint32_t)32);
}

share* put_cons64_gate(Circuit* c, uint64_t val) {
  uint64_t x = val;
  return c->PutCONSGate(x, (uint32_t)64);
}

share* get_zero_share(Circuit* c, int bitlen){
  if (bitlen == 32)
    //return c->PutCONSGate((uint32_t) 0, (uint32_t) 32);
    return put_cons32_gate(c, 0);
  else
    //return c->PutCONSGate((uint64_t) 0, (uint32_t) 64);
    return put_cons64_gate(c, 0);
}

share* divide32(BooleanCircuit* c, share *ina, share *inb) {
    std::vector<uint32_t> ina_wires = ina->get_wires();
    std::vector<uint32_t> inb_wires = inb->get_wires();
    ina_wires.insert(ina_wires.end(), inb_wires.begin(), inb_wires.end());
    std::vector<uint32_t> out_wires = c->PutGateFromFile("../../bin/circ/int_div_32.aby", ina_wires, 64);
    return create_new_share(out_wires, c);
}

template<typename T>
std::vector<T> make_vector(size_t size) {
    return std::vector<T>(size);
}

template <typename T, typename... Args>
auto make_vector(size_t first, Args... sizes) {
    auto inner = make_vector<T>(sizes...);
    return std::vector<decltype(inner)>(first, inner);
}
"""

main_fn = """
int main(int argc, char** argv) {
    e_role role;
    uint32_t bitlen = 32, secparam = 128, nthreads = 1, nrounds = 1;
    uint16_t port = 7766;
    std::string address = "127.0.0.1";
    int32_t test_op = -1;
    e_mt_gen_alg mt_alg = MT_OT;
    
    read_test_options(&argc, &argv, &role, &bitlen, &secparam, &nrounds, &address, &port, &test_op);
    
    seclvl seclvl = get_sec_lvl(secparam);
    
    ABYParty* party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg);
    bool print_stats = true;
    std::cout << "setup_runtime,online_runtime,setup_comm,online_comm,non_lin_ops_a,max_comm_rounds_a,non_lin_ops_b,max_comm_rounds_b" << std::endl;
    for (uint32_t r = 1; r <= nrounds; r++) {
        build_circuit(party, print_stats);
        party->ExecCircuit();
        std::vector<Sharing*>& sharings = party->GetSharings();
        std::cout << party->GetTiming(P_SETUP) << ","
            << party->GetTiming(P_ONLINE) << ","
            << party->GetSentData(P_SETUP)+party->GetReceivedData(P_SETUP) << ","
            << party->GetSentData(P_ONLINE)+party->GetReceivedData(P_ONLINE) << ","
            << sharings[S_ARITH]->GetNumNonLinearOperations() << ","
            << sharings[S_ARITH]->GetMaxCommunicationRounds() << ","
            << sharings[S_BOOL]->GetNumNonLinearOperations() << ","
            << sharings[S_BOOL]->GetMaxCommunicationRounds() << std::endl; 
        party->Reset();
        print_stats = false;
    }
    return 0;
}
"""

build_circuit_prelude = """
void build_circuit(ABYParty* party, bool print_stats) {
    std::vector<Sharing*>& sharings = party->GetSharings();
    ArithmeticCircuit* arith_circ = (ArithmeticCircuit*) sharings[S_ARITH]->GetCircuitBuildRoutine();
    BooleanCircuit* yao_circ = (BooleanCircuit*) sharings[S_YAO]->GetCircuitBuildRoutine();
    BooleanCircuit* bool_circ = (BooleanCircuit*) sharings[S_BOOL]->GetCircuitBuildRoutine();
"""

build_circuit_conclusion = """
    if (print_stats) {
        std::cout << "A: " << arith_circ->GetNumGates() << std::endl;
        std::cout << "A-MUL: " << ((ArithmeticCircuit *)arith_circ)->GetNumMULGates() << std::endl;
        std::cout << "B: " << bool_circ->GetNumGates() << std::endl;
        std::cout << "B-AND: " << ((BooleanCircuit *)bool_circ)->GetNumANDGates() << std::endl;
        std::cout << "B-XOR: " << ((BooleanCircuit *)bool_circ)->GetNumXORGates() << std::endl;
        std::cout << "Y: " << yao_circ->GetNumGates() << std::endl;
        std::cout << "Y-AND: " << ((BooleanCircuit *)yao_circ)->GetNumANDGates() << std::endl;
        std::cout << "Y-XOR: " << ((BooleanCircuit *)yao_circ)->GetNumXORGates() << std::endl;
    }
}
"""

class InputInfo:
    def __init__(self, is_user_input: bool = False, is_input: bool = False, role: Role = None):
        self.is_user_input = is_user_input
        self.is_input = is_input
        self.role = role


def get_input_info(n: ast.expr) -> InputInfo:
    class InputFinder(ast.NodeVisitor):
        def __init__(self):
            self.info = InputInfo()

        def visit_Call(self, call: ast.Call):
            fn = call.func.id
            if fn == 'Input':
                self.info.is_user_input = True
            elif fn == 'sintarray' or fn == 'sint':
                self.info.is_input = True
            self.generic_visit(call)

        def visit_Attribute(self, attr: ast.Attribute):
            if attr.attr == Role.SERVER.value:
                self.info.role = Role.SERVER.value
            elif attr.attr == Role.CLIENT.value:
                self.info.role = Role.CLIENT.value

    ifinder = InputFinder()
    ifinder.visit(n)
    return ifinder.info


def get_dimensions(n: ast.expr):
    class DimensionFinder(ast.NodeVisitor):
        def __init__(self):
            self.dim = []

        def visit_Call(self, call: ast.Call):
            fn = call.func.id
            if fn == 'sintarray':
                self.dim = call.args
            else:
                self.generic_visit(call)
    df = DimensionFinder()
    df.visit(n)
    return df.dim


class VarTags:
    def __init__(self):
        self.types = defaultdict(object)

    def type(self, var):
        return self.types[var]


class StatementToCode(ast.NodeVisitor):
    TYPES = {
        sint: "uint32_t",
        cint: "uint32_t",
        sintarray: "uint32_t",
        cbool: "bool",
    }

    OPS = {
        ast.Add: "PutADDGate",
        ast.Sub: "PutSUBGate",
        ast.Mult: "PutMULGate",
        ast.Gt: "PutGTGate",
        ast.Not: "PutINVGate",
        ast.Eq: "PutEQGate",
        ast.BitAnd: "PutANDGate",
        ast.BitOr: "PutORGate",
        ast.LShift: "PutLeftShifterGate",
        ast.RShift: "PutBarrelRightShifterGate",
    }

    COPS = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Div: "/",
        ast.Mult: "*",
        ast.Gt: ">",
        ast.Not: "!",
        ast.Eq: "==",
        ast.BitAnd: "&",
        ast.BitOr: "|",
        ast.LShift: "<<",
        ast.RShift: ">>"
    }

    CIRC = {
        Arithmetic: "arith_circ",
        Yao: "yao_circ",
        Boolean: "bool_circ",
    }

    NC = {
        True: "true",
        False: "false",
    }

    @staticmethod
    def ops(op: ast.expr):
        for op_type, s in StatementToCode.OPS.items():
            if isinstance(op, op_type):
                return s

    @staticmethod
    def c_ops(op: ast.expr):
        for op_type, s in StatementToCode.COPS.items():
            if isinstance(op, op_type):
                return s

    @staticmethod
    def circ(p: Circuit):
        for p_type, s in StatementToCode.CIRC.items():
            if isinstance(p, p_type):
                return s

    def get_dims(self, n: ast.Call):
        class DimGetter(ast.NodeVisitor):
            def __init__(self, s2a: StatementToCode):
                self.dims = []
                self.s2a = s2a

            def visit_Call(self, call: ast.Call):
                fn = call.func.id
                if fn == 'sintarray':
                    self.dims = [self.s2a.get_code(a) for a in call.args]
                self.generic_visit(call)
        dm = DimGetter(self)
        dm.visit(n)
        return dm.dims

    def __init__(self, vars: Dict[str, Dict[Circuit, str]], assignment: Assignment, var_types, f: IO,
                 conv_inits: Dict[str, List[Conversion]]):
        self.vars = vars
        self.clear_vars = set()
        self.assignment = assignment
        self._indent = ""
        self.var_types = var_types
        self.lines = []
        self.f = f
        self.last_protocol = {}
        self.conv_inits = conv_inits

    def indent(self):
        self._indent += "    "

    def unindent(self):
        self._indent = self._indent[:-4]

    def clear_code_lines(self):
        self.lines = []

    def add_code_line(self, code: str):
        self.lines.append("%s%s" % (self._indent, code))

    def get_output_code(self):
        return "\n".join(self.lines)

    def get_output_code_for_line(self, line: str):
        return self._indent + line

    def get_code(self, stmt: ast.expr):
        s2c = StatementToCode(self.vars, self.assignment, self.var_types, self.f, self.conv_inits)
        s2c.clear_vars = self.clear_vars
        s2c.visit(stmt)
        return s2c.get_output_code()

    def visit_NameConstant(self, nc: ast.NameConstant):
        self.clear_code_lines()
        if nc.value in self.NC:
            self.add_code_line(self.NC[nc.value])

    def visit_Num(self, num: ast.Num):
        self.clear_code_lines()
        self.add_code_line("%d" % num.n)

    def visit_Name(self, name: ast.Name):
        self.clear_code_lines()
        if name.id in self.vars:
            if "global__centroids" in name.id:
                logging.info("visit_Name :: %s :: global__centroids: %d - %s", name.id, id(name), name.protocol)
            if hasattr(name, 'protocol'):
                self.add_code_line(self.vars[name.id][name.protocol])
            else:
                self.add_code_line(name.id)
        elif name.id in self.clear_vars:
            self.add_code_line(name.id)

    def _visit_Index(self, idx: ast.Index):
        self.clear_code_lines()
        val = self.get_code(idx.value)
        self.add_code_line("%s" % val)

    def visit_Subscript(self, ss: ast.Subscript):
        self.clear_code_lines()
        value = self.get_code(ss.value)
        slice = self.get_code(ss.slice)
        self.add_code_line("%s[%s]" % (value, slice))

    def visit_IfExp(self, ifexp: ast.IfExp):
        self.clear_code_lines()
        print(ast.dump(ifexp.test))
        test = self.get_code(ifexp.test)
        body = self.get_code(ifexp.body)
        orelse = self.get_code(ifexp.orelse)
        self.add_code_line("%s ? %s : %s" % (test, body, orelse))

    def visit_Call(self, call: ast.Call):
        fn = call.func.id
        if fn == 'mux':
            self.clear_code_lines()
            circ = self.circ(call.protocol)
            names = [self.get_code(a) for a in call.args]
            self.add_code_line("%s->PutMUXGate(%s, %s, %s)" % (circ, names[1], names[2], names[0]))
        elif fn == 'sintarray':
            self.clear_code_lines()
            args = [self.get_code(a) for a in call.args]
            self.add_code_line("make_vector<share *>(%s)" % ", ".join(args))
        elif fn == 'Output':
            self.clear_code_lines()
            circ = self.circ(call.protocol)
            names = [self.get_code(a) for a in call.args]
            self.add_code_line("%s->PutOUTGate(%s, ALL)" % (circ, names[0],))
        else:
            self.generic_visit(call)

    def visit_Compare(self, cmp: ast.Compare):
        self.clear_code_lines()
        assert len(cmp.ops) == 1
        circ = self.circ(cmp.protocol)
        op = self.ops(cmp.ops[0])
        left = self.get_code(cmp.left)
        right = self.get_code(cmp.comparators[0])
        #print(ast.dump(cmp), cmp.left.type, cmp.comparators[0].type)
        if is_secret(cmp.type):
            if not is_secret(cmp.left.type):
                left = "put_cons%d_gate(%s, %s)" % (cmp.protocol.bitlen, circ, left)
                #left = "%s->PutCONSGate(%s, %d)" % (circ, left, cmp.protocol.bitlen)
            if not is_secret(cmp.comparators[0].type):
                right = "put_cons%d_gate(%s, %s)" % (cmp.protocol.bitlen, circ, right)
                #right = "%s->PutCONSGate(%s, %d)" % (circ, right, cmp.protocol.bitlen)
        self.add_code_line("%s->%s(%s, %s)" % (circ, op, left, right))

    def visit_UnaryOp(self, uop: ast.UnaryOp):
        print(ast.dump(uop))
        self.clear_code_lines()
        operand = self.get_code(uop.operand)
        if is_secret(uop.type):
            circ = self.circ(uop.protocol)
            op = self.ops(uop.op)
            self.add_code_line("%s->%s(%s)" % (circ, op, operand))
        else:
            op = self.c_ops(uop.op)
            self.add_code_line("%s%s" % (op, operand))

    def visit_BinOp(self, binop: ast.BinOp):
        self.clear_code_lines()
        left = self.get_code(binop.left)
        right = self.get_code(binop.right)
        print(astor.to_source(binop))
        if is_secret(binop.type):
            circ = self.circ(binop.protocol)
            op = self.ops(binop.op)
            if isinstance(binop.op, ast.LShift):
                self.add_code_line("%s->%s(%s, %s)" % (circ, op, left, right))
            elif isinstance(binop.op, ast.Div):
                self.add_code_line("divide%d(%s, %s, %s)" % (binop.protocol.bitlen, circ, left, right))
            else:
                if not is_secret(binop.left.type):
                    left = "put_cons%d_gate(%s, %s)" % (binop.protocol.bitlen, circ, left)
                    #left = "%s->PutCONSGate(%s, %d)" % (circ, left, binop.protocol.bitlen)
                if not is_secret(binop.right.type):
                    right = "put_cons%d_gate(%s, %s)" % (binop.protocol.bitlen, circ, right)
                    #right = "%s->PutCONSGate(%s, %d)" % (circ, right, binop.protocol.bitlen)
                self.add_code_line("%s->%s(%s, %s)" % (circ, op, left, right))
        else:
            op = self.c_ops(binop.op)
            self.add_code_line("%s %s %s" % (left, op, right))

    def visit_Assign(self, ass: ast.Assign):
        ap = None
        if hasattr(ass, 'protocol'):
            ap = ass.protocol
        logging.info("assign: %s, protocol: %s", astor.to_source(ass), ap)
        #logging.info("assign: %s", astor.to_source(ass))
        self.clear_code_lines()
        right = self.get_code(ass.value)
        val_info = get_input_info(ass.value)
        for t in ass.targets:
            var_t = self.var_types[t.base_id]
            if not is_secret(var_t):
                left = self.get_code(t)
                self.add_code_line("%s = %s;" % (left, right))
                continue
            if not val_info.is_user_input and not val_info.is_input:
                if "global__centroids" in t.base_id:
                    logging.info("%s", ass.protocol)
                left = self.get_code(t)
                self.add_code_line("%s = %s;" % (left, right))
                continue
            if var_t == sintarray:
                if isinstance(t, ast.Subscript):
                    if hasattr(ass, 'protocol'):
                        circ = self.circ(ass.protocol)
                        bitlen = ass.protocol.bitlen
                    else:
                        circ = self.circ(self.last_protocol[get_baseid(t)])
                        bitlen = self.last_protocol[get_baseid(t)].bitlen
                    left = self.get_code(t)
                    self.add_code_line("%s = put_cons%d_gate(%s, %s);" % (left, bitlen, circ, right))
                    continue
                self.add_code_line("auto %s = %s;" % (t.id, right))
                if t.id in self.conv_inits:
                    for c in self.conv_inits[t.id]:
                        conv_name = "%s_%s" % (t.id, c.to)
                        self.add_code_line("auto %s = %s;" % (conv_name, right))
                if val_info.is_input and not val_info.is_user_input:
                    if hasattr(ass, 'protocol'):
                        v = get_baseid(t)
                        self.last_protocol[v] = ass.protocol
                    continue
                tmp_var_id = "%s_value" % t.id
                self.add_code_line("%s %s;" % (self.TYPES[var_t], tmp_var_id))
                circ = self.circ(ass.protocol)
                dims = self.get_dims(ass.value)
                idx = ""
                for d, loop_var in zip(dims, loop_vars.get_loop_vars(len(dims))):
                    self.add_code_line("for (int %s = 0; %s < %s; %s++) {" % (loop_var, loop_var, d, loop_var))
                    self.indent()
                    idx += "[%s]" % loop_var
                # self.add_code_line("std::cin >> %s;" % tmp_var_id)
                self.add_code_line("%s = 5;" % tmp_var_id)
                self.add_code_line(
                    "%s%s = %s->PutINGate(%s, %d, %s);" % (
                        t.id, idx, circ, tmp_var_id, ass.protocol.bitlen, val_info.role
                    )
                )
                for _ in dims:
                    self.unindent()
                    self.add_code_line("}")
            elif var_t == sint:
                circ = self.circ(ass.value.protocol)
                if val_info.is_input and not val_info.is_user_input:
                    left = self.get_code(t)
                    self.add_code_line("%s = put_cons%d_gate(%s, %s);" % (left, ass.value.protocol.bitlen, circ, right))
                    continue
                tmp_var_id = "%s_value" % t.id
                self.add_code_line("%s %s;" % (self.TYPES[var_t], tmp_var_id))
                #self.add_code_line("%s = %s;" % (tmp_var_id, right))
                #self.add_code_line("std::cin >> %s;" % tmp_var_id)
                self.add_code_line("%s = 5;" % tmp_var_id)
                self.add_code_line(
                    "%s = %s->PutINGate(%s, %d, %s);" % (
                        t.id, circ, tmp_var_id, ass.protocol.bitlen, val_info.role
                    )
                )
        print(self.get_output_code(), file=self.f)

    def declare_variables(self, vars: set):
        self.clear_code_lines()
        declared = set()
        for v, p in vars:
            if v in declared:
                continue
            base_var = remove_version(v)
            t = self.var_types[base_var]
            print(base_var, t)
            if is_secret(t):
                for otherp in PROTOCOLS:
                    tp_name = v
                    if p and p != otherp:
                        tp_name = "%s_%s" % (v, otherp)
                    if t == sint:
                        self.add_code_line("share *%s;" % tp_name)
                    elif t == sintarray:
                        pass
                        #self.add_code_line("share **%s;" % tp_name)
                    self.vars[v][otherp] = tp_name
            elif v not in self.clear_vars:
                self.add_code_line("%s %s;" % (self.TYPES[t], v))
                self.clear_vars.add(v)
            declared.add(v)
        print(self.get_output_code(), file=self.f)


def get_baseid(n: ast.expr):
    if isinstance(n, ast.Name):
        return remove_version(n.id)
        return n.baseid
    elif isinstance(n, ast.Subscript):
        while not isinstance(n, ast.Name):
            n = n.value
        return remove_version(n.id)
        return n.baseid
    print("No baseid for", astor.to_source(n).strip())

def get_name(n: ast.expr):
    if isinstance(n, ast.Name):
        return n.id
    elif isinstance(n, ast.Subscript):
        while not isinstance(n, ast.Name):
            n = n.value
        return n.id


class TypeFinder(ast.NodeTransformer):
    def __init__(self, id_to_type: Dict[str, Union[cint, sintarray]]):
        self.type = None
        self.parent = None
        self.var_types = id_to_type
        self.vars = set()

    def get_type(self, n: ast.expr):
        tf = TypeFinder(self.var_types)
        tf.visit(n)
        n.type = tf.type
        return tf.type

    def visit_IfExp(self, if_exp: ast.IfExp):
        pass

    def visit_Constant(self, con: ast.Constant):
        if isinstance(con.value, int) and self.type is None:
            self.type = cint

    def visit_Num(self, num: ast.Num):
        if self.type is None:
            self.type = cint
        return num

    def visit_NameConstant(self, nc: ast.NameConstant):
        if self.type is None:
            self.type = cbool
        return nc

    def visit_Compare(self, cmp: ast.Compare):
        left_t = self.get_type(cmp.left)
        self.type = left_t
        for c in cmp.comparators:
            c_t = self.get_type(c)
            if is_secret(c_t):
                self.type = c_t
        return cmp

    def visit_Index(self, idx: ast.Index):
        self.get_type(idx.value)

    def visit_Subscript(self, ss: ast.Subscript):
        self.get_type(ss.slice)
        value = ss.value
        while isinstance(value, ast.Subscript):
            value = value.value
        value_t = self.get_type(value)
        if is_secret(value_t):
            self.type = sint
        return ss

    def visit_UnaryOp(self, uop: ast.UnaryOp):
        self.type = self.get_type(uop.operand)
        return uop

    def visit_Call(self, call: ast.Call):
        fn = call.func.id
        if fn == 'cint':
            self.type = cint
            #ret = call.args[0]
            #self.get_type(ret)
        elif fn == 'sint':
            self.type = sint
            #ret = call.args[0]
            #self.get_type(ret)
        elif fn == 'sintarray':
            self.type = sintarray
            #ret = call.args
        for a in call.args:
            a_t = self.get_type(a)
            if self.type is None:
                self.type = a_t
            elif is_secret(a_t):
                self.type = a_t
        return call

    def visit_BinOp(self, binop: ast.BinOp):
        left_t = self.get_type(binop.left)
        right_t = self.get_type(binop.right)
        if is_secret(left_t):
            self.type = left_t
        elif is_secret(right_t):
            self.type = right_t
        else:
            self.type = left_t
        return binop

    def visit_Name(self, name: ast.Name):
        name.base_id = remove_version(name.id)
        #name.id = name.base_id
        if not is_secret(self.type) and name.base_id in self.var_types:
            self.type = self.var_types[name.base_id]
        if hasattr(name, 'protocol'):
            self.vars.add((name.id, name.protocol))
        else:
            self.vars.add((name.id, None))
        return name

    def visit_Assign(self, ass: ast.Assign):
        value_t = self.get_type(ass.value)
        for i, t in enumerate(ass.targets):
            ass.targets[i] = self.visit(t)
            t.base_id = get_baseid(t)
            if t.base_id not in self.var_types:
                self.var_types[t.base_id] = value_t
            elif is_secret(value_t) and not is_secret(self.var_types[t.base_id]):
                self.var_types[t.base_id] = value_t
            print(ast.dump(t))
        return ass



class ABYCodeGenerator:
    def __init__(self, cfg: CFG, assignment: Assignment):
        self.cfg = cfg
        self.assignment = assignment
        self.assignment.tag_ast()

    CONV = {
        A2B: "bool_circ->PutA2BGate(%s, yao_circ)",
        A2Y: "yao_circ->PutA2YGate(%s)",
        B2A: "arith_circ->PutB2AGate(%s)",
        B2Y: "yao_circ->PutB2YGate(%s)",
        Y2B: "bool_circ->PutY2BGate(%s)",
        Y2A: "arith_circ->PutY2AGate(%s, bool_circ)"
    }

    def conv_code(self, c: Circuit, input: str):
        for c_t, s in self.CONV.items():
            if isinstance(c, c_t):
                return s % input

    def generate_code(self):
        f = open("out.cpp", "w")
        print(prelude, file=f)
        tf = TypeFinder({})
        starting_n = None
        for n in self.cfg.graph.nodes:
            if self.cfg.graph.in_degree(n) == 0:
                starting_n = n
                break
        nbrs = [starting_n]
        while nbrs:
            curr = nbrs.pop(0)
            tf.visit(curr.expr)
            for v in self.cfg.graph[curr]:
                nbrs.append(v)
        print(build_circuit_prelude, file=f)
        conv_inits = {}
        for c_node, cs in self.assignment.conversions.items():
            name = get_name(c_node.expr.targets[0])
            conv_inits[name] = cs
        s2a = StatementToCode(defaultdict(dict), self.assignment, tf.var_types, f, conv_inits)
        s2a.indent()
        s2a.declare_variables(tf.vars)
        nbrs = [starting_n]
        iters = 1
        loop_bounds = []
        while nbrs:
            #print(loop_bounds, iters)
            curr = nbrs.pop(0)
            if self.cfg.backedges.in_degree(curr) > 0:
                # Start the for loop
                loop_bound = curr.res_weight / iters
                loop_bounds.append(loop_bound)
                iters = iters * loop_bound
                loop_var = loop_vars.get_loop_var()
                forloop = "%sfor(int LPVAR = 0; LPVAR < %d; LPVAR++) {" % (s2a._indent, loop_bound)
                forloop = forloop.replace("LPVAR", loop_var)
                print(forloop, file=f)
                s2a.indent()
            tf.visit(curr.expr)
            if True or not hasattr(curr.expr, 'pseudo_phi'):
                s2a.visit(curr.expr)

            for c_node, cs in self.assignment.conversions.items():
                for c in cs:
                    if not c.edge:
                        if curr != c_node:
                            continue
                        lhs_target = curr.expr.targets[0]
                    else:
                        if curr != c.edge[0]:
                            continue
                        lhs_target = c_node.expr.targets[0]
                    lhs_name = get_name(lhs_target)
                    lhs = lhs_name
                    conv_name = "%s_%s" % (lhs_name, c.to)
                    conv_lhs = conv_name
                    if isinstance(lhs_target, ast.Subscript):
                        lhs = s2a.get_code(lhs_target)
                        index = lhs[lhs.find('['):]
                        conv_lhs = "%s%s" % (conv_name, index)
                    code = "%s = %s;" % (conv_lhs, self.conv_code(c.type, lhs))
                    print(s2a.get_output_code_for_line(code), file=f)
                    s2a.vars[lhs_name][c.to] = conv_name
            nbrs.extend(self.cfg.graph[curr])
            if self.cfg.backedges.out_degree(curr) > 0:
                s2a.unindent()
                print("%s}" % s2a._indent, file=f)
                iters = iters / loop_bounds.pop()
        print(s2a.vars)
        print(build_circuit_conclusion, file=f)
        print(main_fn, file=f)


class LoopVars:
    BASE = "__lpvar_%d"
    def __init__(self):
        self.ctr = 0

    def get_loop_vars(self, n: int):
        ret = (self.BASE % i for i in range(self.ctr, self.ctr + n))
        self.ctr += n
        return ret

    def get_loop_var(self):
        ret = self.BASE % self.ctr
        self.ctr += 1
        return ret

loop_vars = LoopVars()