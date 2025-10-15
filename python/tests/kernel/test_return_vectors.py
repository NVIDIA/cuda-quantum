import cudaq
import pytest
import os


def testReturnVectorBool():

    @cudaq.kernel
    def return_vec_bool() -> list[bool]:
        ret = [True, False]
        return ret

    res = cudaq.run(return_vec_bool, shots_count=1)
    assert res == [[True, False]]

    @cudaq.kernel
    def return_vec_bool_from_measure() -> list[bool]:
        q = cudaq.qvector(5)
        x(q)
        ret = mz(q)
        return ret

    res = cudaq.run(return_vec_bool_from_measure, shots_count=10)
    assert res == [[True] * 5] * 10

    @cudaq.kernel
    def return_vec_bool_from_measure_mix() -> list[bool]:
        q = cudaq.qvector(5)
        for i in range(5):
            if i % 2 == 0:
                x(q[i])
        ret = mz(q)
        return ret

    res = cudaq.run(return_vec_bool_from_measure_mix, shots_count=10)
    assert res == [[True, False, True, False, True]] * 10


def testReturnVectorInt():

    @cudaq.kernel
    def return_vec_int() -> list[int]:
        ret = [1, 2, 3]
        return ret

    res = cudaq.run(return_vec_int, shots_count=1)
    assert res == [[1, 2, 3]]

    @cudaq.kernel
    def return_vec_int_from_measure() -> list[int]:
        q = cudaq.qvector(5)
        x(q)
        ret = mz(q)
        int_ret = [0 for b in ret]
        i = 0
        for b in ret:
            if b:
                int_ret[i] = 6
            i += 1
        return int_ret

    res = cudaq.run(return_vec_int_from_measure, shots_count=10)
    assert res == [[6] * 5] * 10

    @cudaq.kernel
    def return_vec_int_from_measure_mix() -> list[int]:
        q = cudaq.qvector(5)
        for i in range(5):
            if i % 2 == 0:
                x(q[i])
        ret = mz(q)
        int_ret = [0 for b in ret]
        i = 0
        for b in ret:
            if b:
                int_ret[i] = 6
            i += 1
        return int_ret

    res = cudaq.run(return_vec_int_from_measure_mix, shots_count=10)
    assert res == [[6, 0, 6, 0, 6]] * 10


def testReturnVectorFloat():

    @cudaq.kernel
    def return_vec_float() -> list[float]:
        ret = [1.1, 2.2, 3.3]
        return ret

    res = cudaq.run(return_vec_float, shots_count=1)
    assert res == [[1.1, 2.2, 3.3]]

    @cudaq.kernel
    def return_vec_float_from_measure() -> list[float]:
        q = cudaq.qvector(5)
        x(q)
        ret = mz(q)
        float_ret = [0.0 for b in ret]
        i = 0
        for b in ret:
            if b:
                float_ret[i] = 6.6
            i += 1
        return float_ret

    res = cudaq.run(return_vec_float_from_measure, shots_count=10)
    assert res == [[6.6] * 5] * 10

    @cudaq.kernel
    def return_vec_float_from_measure_mix() -> list[float]:
        q = cudaq.qvector(5)
        for i in range(5):
            if i % 2 == 0:
                x(q[i])
        ret = mz(q)
        float_ret = [0.0 for b in ret]
        i = 0
        for b in ret:
            if b:
                float_ret[i] = 6.6
            i += 1
        return float_ret

    res = cudaq.run(return_vec_float_from_measure_mix, shots_count=10)
    assert res == [[6.6, 0.0, 6.6, 0.0, 6.6]] * 10


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
