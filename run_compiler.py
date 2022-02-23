import ast
import sys

from compiler import transform

in_file = sys.argv[1]
out_file = sys.argv[2]
with open(in_file, "r") as f:
    node = ast.parse(f.read())
    node = transform(node, out_file)
