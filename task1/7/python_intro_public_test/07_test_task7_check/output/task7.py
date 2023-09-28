def find_modified_max_argmax(L, f):
    L = [f(x) for x in L if type(x) is int]
    return (max(L), L.index(max(L))) if L