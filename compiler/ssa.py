import ast
import astor

from collections import defaultdict
from typing import List, Optional, Dict


def remove_version(var: str):
    return "_".join(var.split(VariableVersions.DELIMITER)[:2])


class VariableVersions:
    DELIMITER = "__"

    def __init__(self, name: str, parent: Optional['VariableVersions'] = None):
        self.dict = defaultdict(int)
        self.parent = parent
        self.name = name
        self.modified = {}
        self.accessed = set()
        if self.parent is not None:
            if self.parent.parent is not None:  # Omit global
                self.name = "%s%s%s" % (self.parent.name, self.DELIMITER, self.name)

    def get_variable_id(self, var: str, version: Optional[int] = None) -> str:
        self.accessed.add(var)
        if self[var] == 0:
            return self.parent.get_variable_id(var, version)
        if version is None:
            version = self[var]
        return self.DELIMITER.join((self.name, var, str(version)))

    def __getitem__(self, item):
        return self.dict[item]

    def __setitem__(self, key, value):
        if self[key] != 0:
            if key not in self.modified:
                self.modified[key] = self[key]
        self.dict[key] = value

    def keys(self):
        return self.dict.keys()

    def clear_modified(self):
        self.modified = {}

    def clear_accessed(self):
        self.accessed = set()


class VariableTracer(ast.NodeVisitor):
    def __init__(self):
        self.accessed = set()
        self.modified = set()

    def visit_Name(self, name: ast.Name):
        if isinstance(name.ctx, (ast.Store, ast.AugStore)):
            self.modified.add(name.id)
        elif isinstance(name.ctx, ast.Load):
            self.accessed.add(name.id)


class SecretTagger(ast.NodeTransformer):
    def __init__(self):
        self.secret_vars = set()
        self.clear_vars = set()
        self.unknown_vars = set()
        self.second_pass = []


    def visit_Assign(self, ass: ast.Assign):
        print("is secret?", astor.to_source(ass).strip())
        sd = SecretDetector(self.secret_vars, self.clear_vars, self.unknown_vars)
        sd.visit(ass.value)
        print(sd.secret)
        if sd.unknown or sd.unknown_var:
            if not sd.secret:
                self.second_pass.append(ass)
        for t in ass.targets:
            self.sd = sd
            self.visit(t)
            t.is_secret = sd.secret
        return ass

    def visit_Index(self, idx: ast.Index):
        return idx

    def visit_Name(self, name: ast.Name):
        sd = self.sd
        if sd.unknown and not sd.secret:
            self.unknown_vars.add(name.id)
        elif name.id in self.unknown_vars:
            self.unknown_vars.remove(name.id)
        if sd.secret:
            if name.id in self.clear_vars:
                self.clear_vars.remove(name.id)
            self.secret_vars.add(name.id)
        else:
            self.clear_vars.add(name.id)
        return name

    def visit_Module(self, mod: ast.Module):
        for i, stmt in enumerate(mod.body):
            mod.body[i] = self.visit(stmt)
        print("Pass 2")
        while len(self.second_pass) > 0:
            second_pass = self.second_pass
            print(len(self.second_pass))
            self.second_pass = []
            for stmt in second_pass:
                print(astor.to_source(stmt).strip())
                self.visit(stmt)
        return mod


class SecretDetector(ast.NodeVisitor):
    def __init__(self, secret_vars: set, clear_vars: set, unknown_vars: set):
        self.secret = False
        self.unknown_var = False
        self.unknown = False
        self.secret_vars = secret_vars
        self.clear_vars = clear_vars
        self.unknown_vars = unknown_vars

    def visit_Constant(self, cnst: ast.Constant):
        if self.secret is None:
            self.secret = False

    def visit_Name(self, name: ast.Name):
        self.visited_Name = True
        if name.id in self.unknown_vars:
            self.unknown_var = True
            print("Unknown is_secret", name.id)
        if name.id == 'cint' or name.id in self.clear_vars:
            if self.secret is None:
                self.secret = False
        elif name.id == 'sintarray' or name.id == 'sint' or name.id in self.secret_vars:
            self.secret = True
        else:
            self.unknown = True
            print("Unknown is_secret", name.id)


class AnnotationRemover(ast.NodeTransformer):
    def visit_Call(self, call: ast.Call):
        fn = call.func.id
        if fn == 'cint' or fn == 'sint':
            return call.args[0]
        return call


class SSATransformer(ast.NodeTransformer):
    cond_var_name = "cond"
    mux_var_name = "mux"

    def __init__(self):
        self.curr_variable_versions = VariableVersions("global")

    def visit_FunctionDef(self, func_def: ast.FunctionDef):
        self.curr_variable_versions = VariableVersions(func_def.name, parent=self.curr_variable_versions)
        for arg in func_def.args.args:
            self.curr_variable_versions[arg.arg] += 1
            arg.arg = self.curr_variable_versions.get_variable_id(arg.arg)
        func_def.body = self.visit_multiple_stmts(func_def.body)
        self.curr_variable_versions = self.curr_variable_versions.parent
        self.curr_variable_versions[func_def.name] += 1
        return func_def

    def visit_Call(self, call: ast.Call):
        call.args = self.visit_multiple_stmts(call.args)
        if call.func.id == "Output":
            call.is_output = True
        return call

    def visit_Attribute(self, attr: ast.Attribute):
        return attr

    def visit_If(self, if_stmt: ast.If):
        if_stmt.test = self.visit(if_stmt.test)
        cond_name = self.visit(ast.Name(id=self.cond_var_name, ctx=ast.Store()))
        cond = ast.Assign(targets=[cond_name], value=if_stmt.test)
        self.curr_variable_versions.clear_modified()
        if_stmt.body = self.visit_multiple_stmts(if_stmt.body)
        body_modified = self.curr_variable_versions.modified
        self.curr_variable_versions.clear_modified()
        if_stmt.orelse = self.visit_multiple_stmts(if_stmt.orelse)
        orelse_modified = self.curr_variable_versions.modified
        self.curr_variable_versions.clear_modified()
        ret = [cond] + if_stmt.body + if_stmt.orelse
        phis = []
        for var in body_modified.keys() | orelse_modified.keys():
            phi = [var]
            if var in orelse_modified:
                phi.append(self.curr_variable_versions.get_variable_id(var, orelse_modified[var]))
            elif var in body_modified:
                phi.append(self.curr_variable_versions.get_variable_id(var, body_modified[var]))
            phi.append(self.curr_variable_versions.get_variable_id(var))
            phis.append(phi)
        for var, arg1, arg2 in phis:
            self.curr_variable_versions[var] += 1
            var_name = self.curr_variable_versions.get_variable_id(var)
            cond_name = self.curr_variable_versions.get_variable_id(self.cond_var_name)
            args = [
                ast.Name(id=cond_name, ctx=ast.Load()),
                ast.Name(id=arg1, ctx=ast.Load()),
                ast.Name(id=arg2, ctx=ast.Load()),
            ]
            value = ast.Call(func=ast.Name(self.mux_var_name, ctx=ast.Load()), args=args, keywords=[])
            assign = ast.Assign(targets=[ast.Name(id=var_name, ctx=ast.Store())], value=value)
            assign.is_mux = True
            ret.append(assign)
        return ret

    def visit_For(self, for_stmt: ast.For):
        first_name = "%s_first" % self.curr_variable_versions.name
        first = ast.Assign(
            targets=[ast.Name(id=first_name, ctx=ast.Store())],
            value=ast.NameConstant(value=True),
        )
        first.pseudo_phi = True
        for_stmt.target = self.visit(for_stmt.target)
        for_stmt.iter = self.visit(for_stmt.iter)
        tracer = VariableTracer()
        for stmt in for_stmt.body:
            tracer.visit(stmt)
        loop_vars = (tracer.accessed & tracer.modified) & self.curr_variable_versions.keys()
        before_loop_var_names = []
        preamble = []
        for var in loop_vars:
            assign = ast.Assign(targets=[ast.Name(id=var, ctx=ast.Store())], value=ast.Num(n=1)) # placeholder value
            preamble.append(assign)
            before_loop_var_names.append(self.curr_variable_versions.get_variable_id(var))
        for_stmt.body = preamble + for_stmt.body
        for_stmt.body = self.visit_multiple_stmts(for_stmt.body)
        for var, assign, before_loop_var in zip(loop_vars, preamble, before_loop_var_names):
            value = ast.IfExp(
                test=ast.Name(id=first_name, ctx=ast.Load()),
                body=ast.Name(id=before_loop_var, ctx=ast.Load()),
                orelse=ast.Name(id=self.curr_variable_versions.get_variable_id(var), ctx=ast.Load()),
            )
            assign.value = value
            assign.pseudo_phi = True
        not_first = ast.Assign(
            targets=[ast.Name(id=first_name, ctx=ast.Store())],
            value=ast.NameConstant(value=False),
        )
        not_first.pseudo_phi = True
        for_stmt.body.append(not_first)
        for_stmt.orelse = self.visit_multiple_stmts(for_stmt.orelse)
        return [first, for_stmt]

    def visit_Name(self, name: ast.Name) -> ast.Name:
        if isinstance(name.ctx, (ast.Store, ast.AugStore)):
            self.curr_variable_versions[name.id] += 1
        name.id = self.curr_variable_versions.get_variable_id(name.id)
        return name

    def visit_BinOp(self, binop: ast.BinOp):
        binop.left = self.visit(binop.left)
        binop.right= self.visit(binop.right)
        return binop

    def visit_Assign(self, ass: ast.Assign) -> ast.Assign:
        ass.value = self.visit(ass.value)
        if hasattr(ass.value, 'is_output'):
            ass.is_output = ass.value.is_output
        for i, target in enumerate(ass.targets):
            renamed_target = self.visit(target)
            ass.targets[i] = renamed_target
        return ass

    #def visit_Subscript(self, ss: ast.Subscript):
    #    if isinstance(ss.ctx, (ast.Store, ast.AugStore)) and isinstance(ss.value,
    #                                                                    ast.Name):  # TODO: This is a hack, fix it
    #            self.curr_variable_versions[ss.value.id] += 1
    #            ss.value.id = self.curr_variable_versions.get_variable_id(ss.value.id)
    #    else:
    #        ss.value = self.visit(ss.value)
    #    ss.slice = self.visit(ss.slice)
    #    return ss

    def visit_multiple_stmts(self, stmts: List[ast.Expr]) -> List[ast.Expr]:
        ret = []
        for stmt in stmts:
            visit_ret = self.visit(stmt)
            # TODO: Fix this by normalizing the types returned by visits methods.
            if isinstance(visit_ret, list):
                ret.extend(visit_ret)
            else:
                ret.append(visit_ret)
        return ret


class FunctionGatherer(ast.NodeVisitor):
    def __init__(self):
        self.functions = {}

    def visit_FunctionDef(self, function_def: ast.FunctionDef):
        self.functions[function_def.name] = function_def

import copy
class Inliner(ast.NodeTransformer):
    def __init__(self, functions: Dict[str, ast.FunctionDef]):
        self.functions = functions
        self.call_no = defaultdict(int)

    def visit_Assign(self, ass: ast.Assign):
        ret = []
        if isinstance(ass.value, ast.Call):
            call = ass.value
            func_id = call.func.id
            if func_id in self.functions:
                self.call_no[func_id] += 1
                func_def = copy.deepcopy(self.functions[func_id])
                func_filler = FunctionArgFiller(func_id, func_def.args, call.args, self.call_no[func_id])
                func_def = func_filler.visit(func_def)
                ret.extend(func_def.body)
                ass.value = func_filler.ret.value
        ret.append(ass)
        return ret


class FunctionArgFiller(ast.NodeTransformer):
    def __init__(self, func_id: str, function_args: ast.arguments, args: ast.arguments, call_no: int):
        self.func_id = func_id
        self.function_args = function_args
        self.args = args
        self.function_arg_to_arg = {}
        for i, arg in enumerate(self.function_args.args):
            self.function_arg_to_arg[arg.arg] = self.args[i]
        self.call_no = call_no
        self.modified = set()

    def visit_Return(self, ret: ast.Return):
        self.visit(ret.value)
        self.ret = ret
        return None

    def visit_Name(self, name: ast.Name):
        if isinstance(name.ctx, ast.Load) and name.id in self.function_arg_to_arg:
            return copy.deepcopy(self.function_arg_to_arg[name.id])
        if self.func_id in name.id and name.id not in self.modified:
            name.id = "c%d_%s" % (self.call_no, name.id)
            self.modified.add(name.id)
        return name


class Desugarer(ast.NodeTransformer):
    def _visit_compare(self, left: ast.expr, ops: List[ast.cmpop], comparators: List[ast.expr]):
        operands = []
        for op, cmp in zip(ops, comparators):
            if isinstance(op, ast.Gt):
                operands.append(ast.Compare(left=left, ops=[ast.Gt()], comparators=[cmp]))
            elif isinstance(op, ast.GtE):
                operands.append(ast.UnaryOp(op=ast.Not(), operand=ast.Compare(left=cmp, ops=[ast.Gt()], comparators=[left])))
            elif isinstance(op, ast.Lt):
                operands.append(ast.Compare(left=cmp, ops=[ast.Gt()], comparators=[left]))
            elif isinstance(op, ast.LtE):
                operands.append(ast.UnaryOp(op=ast.Not(), operand=ast.Compare(left=left, ops=[ast.Gt()], comparators=[cmp])))
            elif isinstance(op, ast.NotEq):
                operands.append(ast.UnaryOp(op=ast.Not(), operand=ast.Compare(left=left, ops=[ast.Eq()], comparators=[cmp])))
            elif isinstance(op, ast.Eq):
                operands.append(ast.Compare(left=left, ops=[ast.Eq()], comparators=[cmp]))
        return operands


    def visit_Compare(self, cmp: ast.Compare):
        operands = self._visit_compare(cmp.left, cmp.ops, cmp.comparators)
        if len(operands) == 1:
            return operands[0]
        return ast.BoolOp(op=ast.And(), values=operands)



class FunctionRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, _: ast.FunctionDef):
        return
