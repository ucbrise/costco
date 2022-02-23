from compiler.types import cint, sint, sintarray, Input, Role, Output

LEN_A = 1024
LEN_B = 32

a = Input(Role.CLIENT, sintarray(LEN_A))
b = Input(Role.SERVER, sintarray(LEN_B))

def contains(haystack, needle):
	result = sint(0)
	i = cint(0)
	for _ in range(LEN_A):
		equality = haystack[i] == needle
		result = result
		if equality:
			result = sint(1)
	return result

res = sintarray(LEN_B)
i = cint(0)
for _ in range(LEN_B):
	res[i] = contains(a, b[i])
	i = i + 1

out_res = sintarray(LEN_B)
j = cint(0)
for _ in range(LEN_B):
	out_res[j] = Output(res[j])
	j = j + 1


#	public static void main(String[] args) {
#		MPCAnnotation mpc = MPCAnnotationImpl.v();
#
#		int[] pset1 = new int[SIZE1];
#		for(int i = 0; i < SIZE1; i++) {
#			pset1[i] = mpc.IN();
#		}
#		int[] pset2 = new int[SIZE2];
#		for(int i = 0; i < SIZE2; i++) {
#			pset2[i] = mpc.IN();
#		}
#
#		int[] intersection = new int[SIZE1];
#		for(int i = 0; i < SIZE1; i++) {
#			int result = contains(pset2, pset1[i], SIZE2);
#			intersection[i] = result;
#		}
#
#		// alice learns the output
#		for(int i = 0; i < SIZE1; i++) {
#			mpc.OUT(intersection[i]);
#		}
