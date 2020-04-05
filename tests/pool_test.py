# Test
import numpy

from dpsutil.vector.pool import VectorPool

if __name__ == '__main__':
    pool = VectorPool(512, 100000)
    x = numpy.random.rand(100000, 512).astype(numpy.float32)
    pool.add(x)
    assert numpy.all(pool.vectors() == x)

    pool.remove(range(10, 9990))
    pool.remove(range(10, 90010))
    assert numpy.all(pool[10:20] == x[99990:100000])

    pool.clear()
    assert pool.vectors().shape[0] == 0

    pool.add(x)
    assert numpy.all(pool.pop(0) == x[0])
    assert numpy.all(pool[0] == x[1])

    assert numpy.all(pool.get(1) == x[2])

    x = numpy.random.rand(10, 512).astype(numpy.float32)
    pool[:10] = x
    assert numpy.all(pool[:10] == x)

    pool.close()
    pool = VectorPool(512)
    pool.add(x)
    pool.save("test_save", over_write=True)

    pool.close()
    pool = VectorPool(512)
    pool.load("test_save.vecp")
    assert numpy.all(pool.vectors() == x)

    pool.close()
    pool = VectorPool(512)
    pool.insert(range(10), x)

    assert numpy.all(pool.vectors() == x)

    print("Test Pool OK!")
