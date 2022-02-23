import ast
import astor
import logging
import matplotlib.pyplot as plt
import networkx as nx

from .ssa import SSATransformer, FunctionGatherer, Inliner, FunctionRemover, AnnotationRemover, SecretTagger, Desugarer
from .mpc_source import MPCSource, CFG, DefUseChains, Assigner
from .codegen import ABYCodeGenerator

logging.basicConfig(level=logging.INFO)

def transform(mod: ast.Module) -> ast.Module:
    print(ast.dump(mod))
    ds = Desugarer()
    mod = ds.visit(mod)
    print(astor.to_source(mod))
    ssa = SSATransformer()
    mod.body = ssa.visit_multiple_stmts(mod.body)
    function_gatherer = FunctionGatherer()
    function_gatherer.visit(mod)
    inliner = Inliner(function_gatherer.functions)
    mod = inliner.visit(mod)
    function_remover = FunctionRemover()
    mod = function_remover.visit(mod)
    print(astor.to_source(mod))
    print(ast.dump(mod))
    sd = SecretTagger()
    mod = sd.visit(mod)
    ar = AnnotationRemover()
    #mod = ar.visit(mod)
    print(astor.to_source(mod))
    cfg = CFG()
    cfg.visit(mod)
    #plt.figure(figsize=(20, 10))
    cfg.draw_graph()
    #plt.show()
    #plt.savefig('cfg.pdf')
    duc = DefUseChains(cfg.graph, cfg.backedges, cfg.combined)
    for d, u in duc.duc_graph.edges:
        continue
        print("MinCut(%s,%s) = %s" % (d, u, duc.min_cut(d, u)))
    ass = Assigner(cfg)
    a = ass.get_optimal_assignment()
    ass.assign_conversions(a)
    acg = ABYCodeGenerator(cfg, a)
    acg.generate_code()
    return
    mpc = MPCSource()
    mpc.visit(mod)
    mpc.resolve_weights()
    pos = nx.kamada_kawai_layout(mpc.graph)
    plt.figure(figsize=(20, 10))
    nx.draw_networkx_nodes(mpc.graph, pos=pos, node_color="white")
    nx.draw_networkx_edges(mpc.graph, pos=pos, edgelist=mpc.get_forward_edges(), edge_color='gray', width=3)
    nx.draw_networkx_edges(mpc.graph, pos=pos, edgelist=mpc.get_backedges(), edge_color='orange', width=2)
    nx.draw_networkx_labels(mpc.graph, pos=pos, labels=mpc.get_node_labels(), font_size=10, font_color="green")
    #nx.draw_networkx_edge_labels(mpc.graph, pos=pos, edge_labels=mpc.get_edge_labels(), font_size=8, font_color="red")
    plt.show()
    return mod
