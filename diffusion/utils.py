def convert_list_to_shape(vals, t, x_shape):
    vals = vals[t]
    return vals.reshape(len(t), *((1,) * (len(x_shape) - 1)))
