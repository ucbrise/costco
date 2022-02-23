import ast
import astor
import copy
import logging
import networkx as nx
import textwrap

from collections import namedtuple, defaultdict
from random import random, seed
from typing import Dict, List, Tuple, Optional
from .protocols import Arithmetic, Boolean, Yao, Circuit, A2Y, B2Y, B2A, A2B, Y2A, Y2B, CostTotals, CostType

seed(12345)

class Node:
    def __init__(self, expr: ast.expr, weight: ast.expr):
        self.expr = expr
        self.weight = weight
        self.res_weight = 0
        self.depth = 0
        self.sharing = None

    def __repr__(self):
        return astor.to_source(self.expr).strip()
        if isinstance(self.expr, ast.Assign):
            return astor.to_source(self.expr.targets[0]).strip()
        return "Node(expr=%s, weight=%s)" % (
            astor.to_source(self.expr).strip(),
            astor.to_source(self.weight).strip(),
        )

    def __str__(self):
        return self.__repr__()


class CFG(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.backedges = nx.DiGraph()
        self.prev_node = None
        self.curr_weight = ast.Num(1)

    def add_node(self, n: Node):
        self.graph.add_node(n)
        self.backedges.add_node(n)

    def draw_graph(self):
        self.combined = self.graph.copy()
        self.combined.add_edges_from(self.backedges.edges)
        pos = nx.kamada_kawai_layout(self.combined)
        nx.draw_networkx_nodes(self.graph, pos=pos, node_color="white")
        nx.draw_networkx_edges(self.graph, pos=pos, edgelist=self.graph.edges, edge_color='gray', width=4)
        nx.draw_networkx_edges(self.graph, pos=pos, edgelist=self.backedges.edges, edge_color='orange', width=4)
        duc = DefUseChains(self.graph, self.backedges, self.combined)
        nx.draw_networkx_labels(self.graph, pos=pos, labels=self.get_secret_node_labels(), font_size=10, font_color="purple", font_weight="bold")
        nx.draw_networkx_labels(self.graph, pos=pos, labels=self.get_pseudo_phi_labels(), font_size=10, font_color="black", font_weight="bold")
        nx.draw_networkx_labels(self.graph, pos=pos, labels=self.get_clear_node_labels(), font_size=10, font_color="green", font_weight="bold")
        nx.draw_networkx_edges(self.graph, pos=pos, edgelist=duc.get_forward_chains(), edge_color='blue', style="dotted", width=2)
        nx.draw_networkx_edges(self.graph, pos=pos, edgelist=duc.get_backward_chains(), edge_color='red', style="dotted", width=2)

    def get_secret_nodes(self) -> List[Node]:
        secret_nodes = []
        nodes = nx.topological_sort(self.graph)
        for node in nodes:
            if not isinstance(node.expr, ast.Assign):
                continue
            t = node.expr.targets[0]
            if t.is_secret:
                if True or not hasattr(node.expr, 'pseudo_phi'):
                    secret_nodes.append(node)
        return secret_nodes

    def get_secret_node_labels(self) -> Dict[Node, str]:
        return self._get_node_labels(self.get_secret_nodes())

    def get_pseudo_phi_labels(self) -> Dict[Node, str]:
        ret = []
        for node in self.graph.nodes:
            if not isinstance(node.expr, ast.Assign):
                continue
            t = node.expr.targets[0]
            if t.is_secret:
                if hasattr(node.expr, 'pseudo_phi') and node.expr.pseudo_phi:
                    ret.append(node)
        return self._get_node_labels(ret)

    def get__node_labels(self) -> Dict[Node, str]:
        secret_nodes = []
        for node in self.graph.nodes:
            if not isinstance(node.expr, ast.Assign):
                continue
            t = node.expr.targets[0]
            if t.is_secret:
                secret_nodes.append(node)
        return self._get_node_labels(secret_nodes)

    def get_clear_node_labels(self) -> Dict[Node, str]:
        clear_nodes = []
        for node in self.graph.nodes:
            if not isinstance(node.expr, ast.Assign):
                clear_nodes.append(node)
                continue
            t = node.expr.targets[0]
            if not t.is_secret:
                clear_nodes.append(node)
        return self._get_node_labels(clear_nodes)

    @staticmethod
    def _get_node_labels(nodes: List[Node]) -> Dict[Node, str]:
        labels = {}
        for node in nodes:
            print(ast.dump(node.expr))
            #labels[node] = "%s\n\n%s" % (textwrap.fill(astor.to_source(node.expr), width=30), textwrap.fill(astor.to_source(node.weight), width=30))
            labels[node] = "%s" % textwrap.fill(astor.to_source(node.expr), width=30)
            labels[node] = labels[node][:30]
            labels[node] += "\n\nw=%d, d=%d" % (node.res_weight, node.depth)
        return labels

    def visit_For(self, for_stmt: ast.For):
        old_weight = self.curr_weight
        loop_bound = for_stmt.iter.args[0]
        self.curr_weight = ast.BinOp(left=loop_bound, right=self.curr_weight, op=ast.Mult())
        first_node = None
        for s in for_stmt.body:
            self.visit(s)
            if isinstance(s, ast.For):
                continue
            node = Node(s, self.curr_weight)
            self.add_node(node)
            self.graph.add_edge(self.prev_node, node)
            if first_node is None:
                first_node = node
            self.prev_node = node
        self.backedges.add_edge(node, first_node)
        self.curr_weight = old_weight

    def visit_Module(self, mod: ast.Module):
        for s in mod.body:
            self.visit(s)
            if isinstance(s, ast.For):
                continue
            node = Node(s, self.curr_weight)
            self.add_node(node)
            if self.prev_node:
                self.graph.add_edge(self.prev_node, node)
            self.prev_node = node


class Block:
    def __init__(self, nodes: List[Node], blocks: List['Block'], backedge: Optional[Tuple[Node, Node]] = None):
        self.nodes = set(nodes)
        self.inner_blocks = []
        self.backedge = backedge
        for b in blocks:
            if len(self.nodes & b.nodes) == len(b.nodes):
                self.inner_blocks.append(b)
                self.nodes = self.nodes - b.nodes

    def has_node(self, n: Node):
        if n in self.nodes:
            return True
        for ib in self.inner_blocks:
            if ib.has_node(n):
                return True
        return False

    def has_edge(self, n1: Node, n2: Node):
        if n1 in self.nodes or n2 in self.nodes:
            return True
        has_n1 = False
        has_n2 = False
        for ib in self.inner_blocks:
            if n1 in ib.nodes:
                has_n1 = True
            if n2 in ib.nodes:
                has_n2 = True
        return has_n1 and has_n2

    def get_closest_block(self, n: Node):
        if n in self.nodes:
            return self
        for ib in self.inner_blocks:
            b = ib.get_closest_block(n)
            if b:
                return ib

    def get_closest_enclosing_block(self, n1: Node, n2: Node):
        b1 = self.get_closest_block(n1)
        b2 = self.get_closest_block(n2)
        if b1.has_node(n2):
            return b2
        elif b2.has_node(n1):
            return b1
        else:
            last_block = self
            while True:
                check_ibs = False
                for ib in last_block.inner_blocks:
                    if ib.has_node(n1) and ib.has_node(n2):
                        check_ibs = True
                        last_block = ib
                if not check_ibs:
                    break
            return last_block


class DefCollector(ast.NodeVisitor):
    special = ["mux", "sint", "cint", "sintarray", "Input", "Role", "Output"]
    def __init__(self):
        self.defs = set()

    def visit_Subscript(self, ss: ast.Name):
        if isinstance(ss.ctx, ast.Store):
            self.defs.add(target_name(ss.value))
        self.visit(ss.value)
        self.visit(ss.slice)

    def visit_Name(self, name: ast.Name):
        if isinstance(name.ctx, ast.Load):
            if name.id not in self.special:
                self.defs.add(name.id)


def target_name(n: ast.Expr):
    if isinstance(n, ast.Name):
        return n.id
    elif isinstance(n, ast.Subscript):
        while isinstance(n, ast.Subscript):
            n = n.value
        logging.info("target name: %s", astor.to_source(n).strip())
        return n.id


class DefUseChains():
    def __init__(self, graph: nx.DiGraph, backedges: nx.DiGraph, combined: nx.DiGraph):
        self.graph = graph
        self.backedges = backedges
        self.combined = combined
        self.duc_graph = nx.DiGraph()
        for n in self.graph.nodes:
            self.duc_graph.add_node(n)
        self._add_du_chains()
        self._resolve_weights()
        self._resolve_depth()
        self.block = self._get_block()

    def _add_du_chains(self):
        id_to_nodes = defaultdict(list)
        for n in self.graph.nodes:
            stmt = n.expr
            if not isinstance(stmt, ast.Assign):
                continue
            else:
                for t in stmt.targets:
                    name = target_name(t)
                    if name not in id_to_nodes or isinstance(t, ast.Subscript):
                        id_to_nodes[name].append(n)
        print("id_to_nodes")
        #[print(k, v) for k, v in id_to_nodes.items()]
        for n in self.graph.nodes:
            stmt = n.expr
            if not isinstance(stmt, ast.Assign):
                continue
            dc = DefCollector()
            dc.visit(stmt)
            logging.info("n: %s", n)
            for d in dc.defs:
                #d_node = id_to_nodes[d]
                logging.info("d: %s -- %s", d, id_to_nodes[d])
                d_nodes = id_to_nodes[d]
                d_node = None
                if len(d_nodes) == 1:
                    d_node = d_nodes[0]
                if len(d_nodes) > 1:
                    for _d_node in id_to_nodes[d]:
                        try:
                            if n != _d_node:
                                nx.shortest_path(self.graph, _d_node, n)
                                d_node = _d_node
                        except nx.NetworkXNoPath:
                            continue
                if d_node is not None:
                    path = nx.shortest_path(self.combined, d_node, n)
                    last = d_node
                    backward_chain = False
                    for path_node in path[1:]:
                        if self.backedges.has_edge(last, path_node):
                            backward_chain = True
                        last = path_node
                    self.duc_graph.add_edge(d_node, n, backward_chain=backward_chain)

    def _resolve_weights(self):
        for n in self.duc_graph.nodes:
            mod = self.get_constants()
            mod.body.append(ast.Assign(targets=[ast.Name(id="weight", ctx=ast.Store())], value=n.weight))
            import pprint
            #pprint.pprint(ast.dump(mod))
            l = {"weight": 0}
            exec(astor.to_source(mod), globals(), l)
            n.res_weight = l["weight"]

    def _resolve_depth(self):
        for n in self.duc_graph.nodes:
            depth = 1
            fringe = list(self.duc_graph.edges(n))
            visited = set()
            while fringe:
                d, u = fringe.pop()
                visited.add(u)
                if self.duc_graph[d][u]['backward_chain']:
                    depth = n.res_weight
                nbrs = self.duc_graph.edges(u)
                for d, u in nbrs:
                    if u not in visited:
                        fringe.append((d, u))
            n.depth = depth


    def get_constants(self) -> ast.Module:
        body = []
        for n in self.duc_graph.nodes:
            if self.duc_graph.in_degree(n) == 0:
                expr = n.expr
                if isinstance(expr, ast.Assign):
                    if isinstance(n.expr.value, ast.Num):
                        body.append(n.expr)
                else:
                    body.append(n.expr)
        return ast.Module(body=body)

    def get_forward_chains(self):
        ret = []
        for u, v, d in self.duc_graph.edges(data=True):
            if 'backward_chain' in d and not d['backward_chain']:
                ret.append((u, v))
        return ret

    def get_backward_chains(self):
        ret = []
        for u, v, d in self.duc_graph.edges(data=True):
            if 'backward_chain' in d and d['backward_chain']:
                ret.append((u, v))
        return ret

    def min_cut(self, d: Node, u: Node) -> Tuple[Tuple[Node, Node], int]:
        du_chain = self.duc_graph[d][u]
        b_enc = self.block.get_closest_enclosing_block(d, u)
        if du_chain['backward_chain']:
            return b_enc.backedge, d.res_weight
        else:
            path = nx.shortest_path(self.graph, d, u)
            curr = d
            for v in path[1:]:
                if b_enc.has_edge(curr, v):
                    return (curr, v), u.res_weight
                curr = v
        print("No min cut found... %s, %s" % (d, u))

    def _get_block(self) -> Block:
        starting_n = None
        for n in self.graph.nodes:
            if self.graph.in_degree(n) == 0:
                starting_n = n
                break
        curr = starting_n
        curr_blocks = []
        nodes = []
        while self.graph.out_degree(curr) > 0:
            if self.backedges[curr]:
                v = next(self.backedges.neighbors(curr))
                path = nx.shortest_path(self.graph, v, curr)
                curr_blocks.append(Block(path, curr_blocks, (curr, v)))
            for v in self.graph[curr]:
                curr = v
                nodes.append(v)
        return Block(nodes, curr_blocks, None)


class OpTracker(ast.NodeVisitor):
    def __init__(self):
        self.ops = []
        self.cmp = False
        self.shift = False

    def visit_BinOp(self, binop: ast.BinOp):
        self.ops.append(binop)
        self.visit(binop.left)
        self.visit(binop.right)
        if isinstance(binop.op, (ast.LShift, ast.RShift)):
            self.shift = True

    def visit_Compare(self, cmp: ast.Compare):
        self.ops.append(cmp)
        self.cmp = True


class NodeOpInfo:
    def __init__(self, n: Node, parents: List[Node], ops: Optional[List[Node]] = None, is_mux: bool = False,
                 is_bool: bool = False, is_output: bool = False):
        self.n = n
        if ops is None:
            self.ops = []
        else:
            self.ops = ops
        self.is_mux = is_mux
        self.is_bool = is_bool
        self.is_output = is_output
        self.parents = parents

    def get_cost(self, p: Circuit, total: CostTotals):
        n = self.n.res_weight
        depth = self.n.depth
        if self.is_mux:
            p.get_cost(p.mux(n, depth), total)
        elif self.is_output:
            return
        else:
            for expr in self.ops:
                if isinstance(expr, ast.BinOp):
                    op = expr.op
                    if isinstance(op, ast.Mult):
                        p.get_cost(p.mul(n, depth), total)
                    elif isinstance(op, ast.Add):
                        p.get_cost(p.add(n, depth), total)
                    elif isinstance(op, ast.Sub):
                        p.get_cost(p.sub(n, depth), total)
                elif isinstance(expr, ast.Compare):
                    for op in expr.ops:
                        if isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
                            p.get_cost(p.gt(n, depth), total)
                        elif isinstance(op, (ast.NotEq, ast.Eq)):
                            p.get_cost(p.eq(n, depth), total)


class Conversion(namedtuple('Conversion', ['type', 'to', 'edge', 'weight'])):
    def __eq__(self, other):
        return self.type == other.type and self.edge == other.edge


class Assignment:
    def __init__(self, assignable_node_info: Dict[Node, NodeOpInfo]):
        self.ani = assignable_node_info
        self.protocols = {}     # type: Dict[Node, Circuit]
        self.depth = {}     # type: Dict[Node, int]
        self.conversions = defaultdict(list)    # type: Dict[Node, List[Conversion]]

    def __copy__(self):
        copy = Assignment(self.ani)
        for k, v in self.protocols.items():
            copy.protocols[k] = v
        return copy

    def add_conversion(self, n: Node, conv: Circuit, to: Circuit, edge: Tuple[Node, Node], weight: int):
        c = Conversion(conv, to, edge, weight)
        if c not in self.conversions[n]:
            self.conversions[n].append(c)

    def get_cost(self):
        # TODO: Enforce order when comptuing the cost.
        total = CostTotals()
        for n, p in self.protocols.items():
            if n in self.ani:
                self.ani[n].get_cost(p, total)
            else:
                p.get_input_cost(n.res_weight, total)
        for p in set(self.protocols.values()):
            p.get_one_time_cost(total)
        for n, convs in self.conversions.items():
            for c in convs:
                c.type.get_cost({c.type.OPS[0]: c.weight}, total)
        if total[CostType.MEM] > 10000:
            return total[CostType.RT_MEM_PRESSURE]
        else:
            return total[CostType.RT]

    def tag_ast(self):
        for n, p in self.protocols.items():
            pt = ProtocolTagger(p)
            pt.visit(n.expr)


class ProtocolCounts(defaultdict):
    def __init__(self):
        super(ProtocolCounts, self).__init__(float)


class Assigner:
    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.duc = DefUseChains(cfg.graph, cfg.backedges, cfg.combined)

    def get_assignable_nodes(self) -> (Dict[Node, NodeOpInfo], List[Node]):
        ret = {}
        ordered_keys = []
        secret_nodes = self.cfg.get_secret_nodes()
        for n in secret_nodes:
            parents = [u for u, v in self.duc.duc_graph.in_edges(n)]
            expr = n.expr
            if hasattr(expr, 'is_mux') and expr.is_mux:
                na = NodeOpInfo(n, parents, is_mux=True)
                if n not in ret:
                    ordered_keys.append(n)
                ret[n] = na
                continue
            if hasattr(expr, 'is_output') and expr.is_output:
                na = NodeOpInfo(n, parents, is_output=True)
                if n not in ret:
                    ordered_keys.append(n)
                ret[n] = na
                continue
            ot = OpTracker()
            ot.visit(expr.value)
            if len(ot.ops) > 0:
                print(ot.ops)
                is_bool = ot.cmp or ot.shift
                na = NodeOpInfo(n, parents, ops=ot.ops, is_bool=is_bool)
                if n not in ret:
                    ordered_keys.append(n)
                ret[n] = na
                print("Bool only", is_bool, n)
            else:
                print("Cannot assign", n)
        return ret, ordered_keys

    def get_optimal_assignment(self) -> Assignment:
        assignable_node_info, ordered_ani = self.get_assignable_nodes()
        assignment = Assignment(assignable_node_info)
        print(len(ordered_ani))
        #for n in ordered_ani:
        #    assignment.protocols[n] = yao
        #return assignment
        assignments = self._make_assignments_greedy(ordered_ani, assignment, None)
        #for n in assignable_node_info:
        #    assignment.protocols[n] = yao
        #assignments = [assignment]
        min_cost = None
        min_assignment = None
        from tqdm import tqdm
        for assignment in tqdm(assignments):
            cost = assignment.get_cost()
            if min_cost is not None and cost > min_cost:
                continue
            self.assign_conversions(assignment)
            cost = assignment.get_cost()
            if min_cost is None or cost < min_cost:
                min_cost = cost
                min_assignment = assignment
        import pickle
        with open("min_assignment", "wb") as f:
            pickle.dump(min_assignment, f)
        print("min cost", min_cost)
        for d, a in min_assignment.protocols.items():
            print("assignment -- %s: %s weight=%d" % (d, a, d.res_weight))
        for d, a in min_assignment.conversions.items():
            print("conv -- %s: %s" % (d, a))
        return min_assignment

    def _make_assignments_greedy(self, nodes: List[Node], parent_assignment: Assignment, last_p: Circuit):
        if len(nodes) == 0:
            return [parent_assignment]
        ret = []
        u = nodes[0]
        info = parent_assignment.ani[u]
        if last_p is None:
            if not info.is_mux and not info.is_bool:
                a1 = copy.copy(parent_assignment)
                a1.protocols[u] = arith
                ret.extend(self._make_assignments_greedy(nodes[1:], a1, arith))
            a1 = copy.copy(parent_assignment)
            a1.protocols[u] = bool
            ret.extend(self._make_assignments_greedy(nodes[1:], a1, bool))
            a1 = copy.copy(parent_assignment)
            a1.protocols[u] = yao
            ret.extend(self._make_assignments_greedy(nodes[1:], a1, yao))
            return ret
        curr_p = defaultdict(int)
        for d, _ in self.duc.duc_graph.in_edges(u):
            if d in parent_assignment.protocols:
                p = parent_assignment.protocols[d]
                _, w = self.duc.min_cut(d, u)
                curr_p[p] += w
        last_p_cost = CostTotals()
        if last_p != arith or (not info.is_mux and not info.is_bool):
            info.get_cost(last_p, last_p_cost)
            a1 = copy.copy(parent_assignment)
            a1.protocols[u] = last_p
            ret.extend(self._make_assignments_greedy(nodes[1:], a1, last_p))
        min_p = None
        min_cost = None
        for p in [arith, bool, yao]:
            if p == last_p:
                continue
            if p == arith and (info.is_mux or info.is_bool):
                continue
            c = CostTotals()
            for cp, w in curr_p.items():
                if cp == p:
                    continue
                conv = get_conversion(cp, p)
                conv.get_cost({conv.OPS[0]: w}, c)
            info.get_cost(p, c)
            if min_cost is None or c[CostType.RT] < min_cost:
                min_p = p
                min_cost = c[CostType.RT]
        use_min_p = True
        if last_p_cost[CostType.RT] > 0 and min_cost > last_p_cost[CostType.RT] * 5:
            if random() > 0.2:
                use_min_p = False
        if use_min_p:
            a1 = copy.copy(parent_assignment)
            a1.protocols[u] = min_p
            ret.extend(self._make_assignments_greedy(nodes[1:], a1, min_p))
        return ret

    def __make_assignments(self, nodes: List[Node], parent_assignment: Assignment):
        if len(nodes) == 0:
            return [parent_assignment]
        ret = []
        u = nodes[0]
        info = parent_assignment.ani[u]
        curr_p = None
        weight = 0
        for d, _ in self.duc.duc_graph.in_edges(u):
            if d in parent_assignment.protocols:
                if curr_p is None:
                    curr_p = parent_assignment.protocols[d]
                if curr_p == parent_assignment.protocols[d]:
                    _, w = self.duc.min_cut(d, u)
                    weight += w
        if curr_p is not None:
            c_noconv = CostTotals()
            should_conv = False
            if curr_p == arith and info.is_mux or info.is_bool:
                should_conv = True
            else:
                parent_assignment.ani[u].get_cost(curr_p, c_noconv)
            c_noconv = c_noconv[CostType.RT] * 10
            if curr_p != arith:
                c_conv = CostTotals()
                if not info.is_mux and not info.is_bool:
                    parent_assignment.ani[u].get_cost(arith, c_conv)
                    conv = get_conversion(curr_p, arith)
                    if should_conv or c_conv[CostType.RT] + (conv.costs[conv.OPS[0]][CostType.RT] * weight) < c_noconv:
                            a1 = copy.copy(parent_assignment)
                            a1.protocols[u] = arith
                            ret.extend(self.__make_assignments(nodes[1:], a1))
                            should_conv = True
            if curr_p != bool:
                c_conv = CostTotals()
                parent_assignment.ani[u].get_cost(bool, c_conv)
                conv = get_conversion(curr_p, bool)
                if should_conv or c_conv[CostType.RT] + (conv.costs[conv.OPS[0]][CostType.RT] * weight) < c_noconv:
                    a1 = copy.copy(parent_assignment)
                    a1.protocols[u] = bool
                    ret.extend(self.__make_assignments(nodes[1:], a1))
                    should_conv = True
            if curr_p != yao:
                c_conv = CostTotals()
                parent_assignment.ani[u].get_cost(yao, c_conv)
                conv = get_conversion(curr_p, yao)
                if should_conv or c_conv[CostType.RT] + (conv.costs[conv.OPS[0]][CostType.RT] * weight) < c_noconv:
                    a1 = copy.copy(parent_assignment)
                    a1.protocols[u] = yao
                    ret.extend(self.__make_assignments(nodes[1:], a1))
                    should_conv = True
            if not should_conv:
                a1 = copy.copy(parent_assignment)
                a1.protocols[u] = curr_p
                ret.extend(self.__make_assignments(nodes[1:], a1))
        else:
            if not info.is_mux and not info.is_bool:
                a1 = copy.copy(parent_assignment)
                a1.protocols[u] = arith
                ret.extend(self.__make_assignments(nodes[1:], a1))
            a2 = copy.copy(parent_assignment)
            a2.protocols[u] = yao
            ret.extend(self.__make_assignments(nodes[1:], a2))
            a3 = copy.copy(parent_assignment)
            a3.protocols[u] = bool
            ret.extend(self.__make_assignments(nodes[1:], a3))
        return ret

    def _make_assignments(self, nodes: List[Node], parent_assignment: Assignment):
        if len(nodes) == 0:
            return [parent_assignment]
        ret = []
        n = nodes[0]
        info = parent_assignment.ani[n]
        if not info.is_mux and not info.is_bool:
            a1 = copy.copy(parent_assignment)
            a1.protocols[n] = arith
            ret.extend(self._make_assignments(nodes[1:], a1))
        a2 = copy.copy(parent_assignment)
        a2.protocols[n] = yao
        ret.extend(self._make_assignments(nodes[1:], a2))
        a3 = copy.copy(parent_assignment)
        a3.protocols[n] = bool
        ret.extend(self._make_assignments(nodes[1:], a3))
        return ret

    def assign_conversions(self, a: Assignment):
        secret_nodes = self.cfg.get_secret_nodes()
        curr_pass = secret_nodes
        protocol_counts = defaultdict(lambda: defaultdict(float))
        while curr_pass:
            protocol_counts.clear()
            next_pass = []
            for u in curr_pass:
                if u not in a.protocols:
                    continue
                for d, _ in self.duc.duc_graph.in_edges(u):
                    if d in secret_nodes and d not in a.protocols:
                        _, w = self.duc.min_cut(d, u)
                        p = a.protocols[u]
                        protocol_counts[d][p] = 0
                        if p == bool:
                            conv = get_conversion(arith, p)
                            protocol_counts[d][arith] += conv.costs[conv.OPS[0]][CostType.RT] * w
                            conv = get_conversion(yao, p)
                            protocol_counts[d][yao] += conv.costs[conv.OPS[0]][CostType.RT] * w
                        elif p == arith:
                            conv = get_conversion(bool, p)
                            protocol_counts[d][bool] += conv.costs[conv.OPS[0]][CostType.RT] * w
                            conv = get_conversion(yao, p)
                            protocol_counts[d][yao] += conv.costs[conv.OPS[0]][CostType.RT] * w
                        if p == yao:
                            conv = get_conversion(arith, p)
                            protocol_counts[d][arith] += conv.costs[conv.OPS[0]][CostType.RT] * w
                            conv = get_conversion(bool, p)
                            protocol_counts[d][bool] += conv.costs[conv.OPS[0]][CostType.RT] * w
                    continue
            for d, pc in protocol_counts.items():
                #p = max(pc.keys(), key=lambda x: pc[x])
                p = min(pc.keys(), key=lambda x: pc[x])
                a.protocols[d] = p
                next_pass.append(d)
            curr_pass = next_pass
        for d in secret_nodes:
            for u in self.duc.duc_graph[d]:
                if u not in a.protocols:
                    #a.protocols[u] = a.protocols[d]
                    continue
                if a.protocols[d] == a.protocols[u]:
                    continue
                conv_p = get_conversion(a.protocols[d], a.protocols[u])
                e, w = self.duc.min_cut(d, u)
                a.add_conversion(d, conv_p, a.protocols[u], *self.duc.min_cut(d, u))


class ProtocolTagger(ast.NodeTransformer):
    def __init__(self, p: Circuit):
        self.p = p

    def visit(self, n: ast.expr):
        n.protocol = self.p
        self.generic_visit(n)
        return n


arith = Arithmetic(32)
bool = Boolean(32)
yao = Yao(32)
a2y = A2Y(32)
b2y = B2Y(32)
y2b = Y2B(32)
b2a = B2A(32)
y2a = Y2A(32)
a2b = A2B(32)


def get_conversion(f: Circuit, t: Circuit):
    if f == bool:
        if t == yao:
            return b2y
        return b2a
    elif f == yao:
        if t == bool:
            return y2b
        return y2a
    if t == bool:
        return a2b
    return a2y


class MPCSource(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.curr_nodes = []
        self.nodes = {}
        self.backedge_nodes = []
        self.curr_weight = ast.Num(1)

    def get_constants(self):
        body = []
        for n in self.graph.nodes:
            if self.graph.in_degree(n) == 0:
                if isinstance(n.expr, ast.Assign):
                    body.append(n.expr)
        return ast.Module(body=body)

    def resolve_weights(self):
        for n in self.graph.nodes:
            mod = self.get_constants()
            mod.body.append(ast.Assign(targets=[ast.Name(id="weight", ctx=ast.Store())], value=n.weight))
            l = {"weight": 0}
            exec(astor.to_source(mod), globals(), l)
            #print(l)

    def get_forward_edges(self) -> List[Tuple[Node, Node]]:
        ret = []
        for u, v, d in self.graph.edges(data=True):
            if 'backedge' not in d:
                ret.append((u, v))
        return ret

    def get_backedges(self) -> List[Tuple[Node, Node]]:
        ret = []
        for u, v, d in self.graph.edges(data=True):
            if 'backedge' in d:
                ret.append((u, v))
        return ret

    def get_edge_labels(self) -> Dict[Tuple[Node, Node], str]:
        labels = {}
        for u, v, d in self.graph.edges(data=True):
            if 'w' in d:
                labels[(u, v)] = textwrap.fill(astor.to_source(d['w']), width=15)
        return labels

    def get_node_labels(self) -> Dict[Node, str]:
        labels = {}
        for node in self.graph.nodes:
            labels[node] = "%s\n\n%s" % (textwrap.fill(astor.to_source(node.expr), width=30), textwrap.fill(astor.to_source(node.weight), width=30))
        return labels

    def get_def_nodes(self, expr: ast.expr):
        def_collector = DefCollector()
        def_collector.visit(expr)
        ret = []
        for d in def_collector.defs:
            if d in self.nodes:
                ret.append(self.nodes[d])
            else:
                print("missing", d)
        return ret
        #return (self.nodes[d] for d in def_collector.defs)

    def visit_Name(self, name: ast.Name):
        if isinstance(name.ctx, ast.Store):
            self.curr_nodes.append(name.id)

    def visit_Assign(self, ass: ast.Assign):
        self.old_curr_nodes = []
        self.curr_nodes = []
        for i, target in enumerate(ass.targets):
            self.visit(target)
        node = Node(ass, self.curr_weight)
        for n in self.curr_nodes:
            if n not in self.nodes:
                self.nodes[n] = node
        self.curr_nodes = self.old_curr_nodes
        self.graph.add_node(node)
        if hasattr(ass, 'pseudo_phi'):
            self.backedge_nodes.append(node)
        #else:
        for n in self.get_def_nodes(ass.value):
            self.graph.add_edge(n, node)

    def visit_For(self, for_stmt: ast.For):
        self.curr_nodes = []
        self.visit(for_stmt.target)
        for n in self.curr_nodes:
            node = Node(for_stmt.iter, self.curr_weight)
            #self.graph.add_node(node)
            self.nodes[n] = node
        loop_bound = for_stmt.iter.args[0]
        old_weight = self.curr_weight
        self.curr_weight = ast.BinOp(left=loop_bound, right=self.curr_weight, op=ast.Mult())
        self.old_backedge_nodes = self.backedge_nodes
        self.backedge_nodes = []
        for stmt in for_stmt.body:
            self.visit(stmt)
        for node in self.backedge_nodes:
            for d in self.get_def_nodes(node.expr.value):
                if nx.has_path(self.graph, node, d):
                    self.graph.add_edge(d, node, backedge=True)
                else:
                    self.graph.add_edge(d, node)
        self.backedge_nodes = self.old_backedge_nodes
        self.curr_weight = old_weight


