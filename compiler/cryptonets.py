
def activate_sqr(val):
    return val * val

def max_pooling(vals, res, cols, rows):
	rows_res = rows / 2
	cols_res = cols / 2
	i = cint(0)
	for _ in range(rows_res):
		
void max_pooling(DT *vals, DT *OUTPUT_res, unsigned cols, unsigned rows) {
	unsigned rows_res = rows / 2;
	unsigned cols_res = cols / 2;
	for(unsigned i = 0; i < rows_res; i++) {
		for(unsigned j = 0; j < cols_res; j++) {
			unsigned x = j * 2;
			unsigned y = i * 2;
			DT max = vals[y*cols + x];
			if(vals[y*cols + x + 1] > max) {
				max = vals[y*cols + x + 1];
			}
			if(vals[(y + 1) *cols + x] > max) {
				max = vals[(y + 1) * cols + x];
			}
			if(vals[(y + 1) *cols + x + 1] > max) {
				max = vals[(y + 1) * cols + x + 1];
			}
			OUTPUT_res[i * cols_res + j] = max;
		}
	}
}
