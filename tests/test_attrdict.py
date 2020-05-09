from dpsutil.attrdict import DefaultDict, attribute_dict, default_dict, default_type_dict

a = DefaultDict(a=1, b=1)
b = DefaultDict({'a': 1, 'b': 1})
c = DefaultDict(**{'a': 1, 'b': 1})
d = DefaultDict({'a': 1, 'b': 1}.items())
e = DefaultDict(zip(['a', 'b'], [1, 1]))
f = DefaultDict((('a', 1), ('b', 1)))
g = DefaultDict(['a', 'b'], 1)
h = DefaultDict({'a', 'b'}, 1)

print(a)
assert a == b == c == d == e == f == g == h


@default_type_dict
class ABY:
    a = 2
    b: float = 2


test = ABY()
test.b = 3
assert test.b == 3
del test.b
assert test.b == 2


