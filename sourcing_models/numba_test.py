from numba import njit, typeof, typed, types

d1 = typed.Dict.empty(types.int32,types.int32)
normal_dict={}
normal_dict[1]=[1]

for k, v in normal_dict.items():
    print(k,v[0])
    d1[k] = v[0]
