from compiler.types import cint, sint, sintarray, Input, Role, Output

def relu(val):
    ret = 0
    res = val > 0
    if res:
        ret = val
    return ret

IMAGE_WIDTH = 28
WINDOW_WIDTH = 5
STRIDE = 1
OUTPUT_CHANNELS = 16

IMAGE_CROP = 24
SIZE_CONVOLUTION_1 = cint(576)
MAX_POOLING_WIDTH_1 = cint(12)

IMAGE_WIDTH_2 = cint(12)
MAX_POOLING_SIZE_1 = cint(2304)
IMAGE_CROP_2 = cint(8)
SIZE_KERNELS_2 = cint(25)
SIZE_ALL_KERNELS_2 = cint(400)

SIZE_CONVOLUTION_2 = cint(64)
SIZE_RELU_2 = cint(1024)

MAX_POOLING_WIDTH_2 = cint(4)
MAX_POOLING_SIZE_2 = cint(256)

FULLY_CONNECTED_WIDTH = cint(100)
FINAL_OUTPUT_CHANNELS = cint(10)

SIZE_FC1 = 25600
SIZE_FC2 = 1000
SIZE_IMAGE = 28 * 28
SIZE_CONV = 16 * 576
SIZE_CONV2 = 16 * 64

kernelsL1 = Input(Role.SERVER, sintarray(SIZE_ALL_KERNELS_2))
kernelsL2 = Input(Role.SERVER, sintarray(SIZE_ALL_KERNELS_2))
kernelsFC1 = Input(Role.SERVER, sintarray(SIZE_FC1))
kernelsFC2 = Input(Role.SERVER, sintarray(SIZE_FC2))
image = Input(Role.CLIENT, sintarray(SIZE_IMAGE))

convolution_layer = sintarray(SIZE_CONV)
convolution_relu = sintarray(SIZE_CONV)

def convolution_naive_outputs(image_offset, image, kernels, OUTPUT_layer, image_width, window_size, output_size, stride, conv_width):
    kernel_size = window_size * window_size
    o = cint(0)
    for _ in range(output_size):
        kernels_offset = o * kernel_size
        OUTPUT_layer_offset = o * (conv_width * conv_width)
        x = cint(0)
        for _ in range(conv_width):
            y = cint(0)
            for _ in range(conv_width):
                oPos = x + y * conv_width
                temp = sint(0)
                wy = cint(0)
                for _ in range(window_size):
                    wx = cint(0)
                    for _ in range(window_size):
                        convPos = wx + wy * window_size
                        kernelPos = kernels_offset + convPos
                        temp = temp + kernels[kernelPos] * image[image_offset + (y * stride + wy) * image_width + (x * stride + wx)]
                        wx = wx + 1
                    wy = wy + 1
                OUTPUT_layer[OUTPUT_layer_offset + oPos] = temp
                y = y + 1
            x = x + 1
        o = o + 1
    return 0


def decomposed_relu(input, OUTPUT_res, len_outer, len_inner):
    i = cint(0)
    for _ in range(len_outer):
        offset = i * len_inner
        j = cint(0)
        for _ in range(len_inner):
            pos = offset+j
            val = input[pos]
            cmp = val <= 0
            if cmp:
                val = sint(0)
            OUTPUT_res[pos] = val
            j = j + 1
        i = i + 1
    return 0

def DT_memset(OUTPUT_res, len, val):
    i = cint(0)
    for _ in range(len):
        OUTPUT_res[i] = val
        i = i + 1
    return 0

x = convolution_naive_outputs(0, image, kernelsL1, convolution_layer, 28, 5, 16, 1, 24)
y = decomposed_relu(convolution_layer, convolution_relu, 16, 576)

pooling_layer = sintarray(MAX_POOLING_SIZE_1)

def convolution_naive_outputs2(image_offset, image, kernels, OUTPUT_layer, image_width, window_size, output_size, stride, conv_width):
    kernel_size = window_size * window_size
    o = cint(0)
    for _ in range(output_size):
        kernels_offset = o * kernel_size
        OUTPUT_layer_offset = o * (conv_width * conv_width)
        x = cint(0)
        for _ in range(conv_width):
            y = cint(0)
            for _ in range(conv_width):
                oPos = x + y * conv_width
                temp = sint(0)
                wy = cint(0)
                for _ in range(window_size):
                    wx = cint(0)
                    for _ in range(window_size):
                        convPos = wx + wy * window_size
                        kernelPos = kernels_offset + convPos
                        temp = temp + kernels[kernelPos] * image[image_offset + (y * stride + wy) * image_width + (x * stride + wx)]
                        wx = wx + 1
                    wy = wy + 1
                OUTPUT_layer[OUTPUT_layer_offset + oPos] = temp
                y = y + 1
            x = x + 1
        o = o + 1
    return 0

def decomposed_relu2(input, OUTPUT_res, len_outer, len_inner):
    i = cint(0)
    for _ in range(len_outer):
        offset = i * len_inner
        j = cint(0)
        for _ in range(len_inner):
            pos = offset+j
            val = input[pos]
            cmp = val <= 0
            if cmp:
                val = sint(0)
            OUTPUT_res[pos] = val
            j = j + 1
        i = i + 1
    return 0

def max_pooling_outputs(vals, OUTPUT_res, outputs, cols, rows):
    size = cols * rows
    row_res = rows / 2
    cols_res = cols / 2
    output_size = row_res * cols_res
    o = cint(0)
    for _ in range(outputs):
        input_offset = o * size
        output_offset = o * output_size
        i = cint(0)
        for _ in range(rows / 2):
            j = cint(0)
            for _ in range(cols / 2):
                x = j * 2
                y = i * 2
                loc1 = input_offset + y * cols + x
                loc2 = input_offset + y * cols + x + 1
                z = y + 1
                loc3 = z * cols + input_offset + x
                loc4 = loc3 + 1
                max = vals[loc1]
                newMax = vals[loc2]
                is_gt =  newMax > max
                if is_gt:
                    max = newMax
                newMax = vals[loc3]
                is_gt =  newMax > max
                if is_gt:
                    max = newMax
                newMax = vals[loc4]
                is_gt =  newMax > max
                if is_gt:
                    max = newMax
                OUTPUT_res[output_offset + i * cols_res + j] = max
                j = j + 1
            i = i + 1
        o = o + 1
    return 0

z = max_pooling_outputs(convolution_layer, pooling_layer, 16, 24, 24)

def sum(OUTPUT_agg, agg, add, len):
    i = cint(0)
    for _ in range(len):
        OUTPUT_agg[i] = agg[i] + add[i]
        i = i + 1
    return 0

convolution_layer_2 = sintarray(SIZE_CONV2)
convolution_relu_2 = sintarray(SIZE_CONV2)
z = DT_memset(convolution_layer_2, 1024, sint(0))
o = cint(0)
for _ in range(OUTPUT_CHANNELS):
    convolution_layer_tmp = sintarray(SIZE_CONV2)
    convolution_layer_tmp_2 = sintarray(SIZE_CONV2)
    #image_start_offset = o * 144
    image_start_offset = 0
    z = convolution_naive_outputs2(image_start_offset, image, kernelsL2, convolution_layer_tmp, 12, 5, 16, 1, 8)
    z = sum(convolution_layer_2, convolution_layer_2, convolution_layer_tmp, 1024)
    o = o + 1

z = decomposed_relu2(convolution_layer_2, convolution_relu_2, 16, 64)

def max_pooling_outputs2(vals, OUTPUT_res, outputs, cols, rows):
    size = cols * rows
    row_res = rows / 2
    cols_res = cols / 2
    output_size = row_res * cols_res
    o = cint(0)
    for _ in range(outputs):
        input_offset = o * size
        output_offset = o * output_size
        i = cint(0)
        for _ in range(rows / 2):
            j = cint(0)
            for _ in range(cols / 2):
                x = j * 2
                y = i * 2
                loc1 = input_offset + y * cols + x
                loc2 = input_offset + y * cols + x + 1
                loc3 = input_offset + (y + 1) * cols + x
                loc4 = loc3 + 1
                max = vals[loc1]
                newMax = vals[loc2]
                is_gt = newMax > max
                if is_gt:
                    max = newMax
                newMax = vals[loc3]
                is_gt = newMax > max
                if is_gt:
                    max = newMax
                newMax = vals[loc4]
                is_gt = newMax > max
                if is_gt:
                    max = newMax
                OUTPUT_res[output_offset + i * cols_res + j] = max
                j = j + 1
            i = i + 1
        o = o + 1
    return 0

pooling_layer_2 = sintarray(MAX_POOLING_SIZE_2)
z = max_pooling_outputs2(convolution_relu_2, pooling_layer_2, 16, 8, 8)

def mmulT_unrolled(a, b, OUTPUT_res, cols_a, cols_b, common):
    i = cint(0)
    for _ in range(cols_a):
        a_offset = i * common
        j = cint(0)
        for _ in range(cols_b):
            b_offset = j * common
            mults = sintarray(common)
            k = cint(0)
            for _ in range(common):
                mults[k] = a[a_offset + k] * b[b_offset + k]
                k = k + 1
            sum = mults[0]
            k = cint(0)
            for _ in range(common):
                sum = sum + mults[k]
                k = k + 1
            OUTPUT_res[i*cols_b+j] = sum
            j = j + 1
        i = i + 1
    return 0

fc_layer = sintarray(FULLY_CONNECTED_WIDTH)
z = mmulT_unrolled(kernelsFC1, pooling_layer_2, fc_layer, 100, 1, 256)

def decomposed_relu3(input, OUTPUT_res, len_outer, len_inner):
    i = cint(0)
    for _ in range(len_outer):
        offset = i * len_inner
        j = cint(0)
        for _ in range(len_inner):
            pos = offset+j
            val = input[pos]
            cmp = val <= 0
            if cmp:
                val = sint(0)
            OUTPUT_res[pos] = val
            j = j + 1
        i = i + 1
    return 0

fc_relu = sintarray(FULLY_CONNECTED_WIDTH)
z = decomposed_relu3(fc_layer, fc_relu, 100, 1)

def mmulT_unrolled2(a, b, OUTPUT_res, cols_a, cols_b, common):
    i = cint(0)
    for _ in range(cols_a):
        a_offset = i * common
        j = cint(0)
        for _ in range(cols_b):
            b_offset = j * common
            mults = sintarray(common)
            k = cint(0)
            for _ in range(common):
                mults[k] = a[a_offset + k] * b[b_offset + k]
                k = k + 1
            sum = mults[0]
            k = cint(0)
            for _ in range(common):
                sum = sum + mults[k]
                k = k + 1
            OUTPUT_res[i*cols_b+j] = sum
            j = j + 1
        i = i + 1
    return 0

output_final_layer = sintarray(FINAL_OUTPUT_CHANNELS)
z = mmulT_unrolled2(kernelsFC2, fc_layer, output_final_layer, 10, 1, 100)

i = cint(0)
for _ in range(10):
    zz = Output(output_final_layer[i])
    i = i + 1
