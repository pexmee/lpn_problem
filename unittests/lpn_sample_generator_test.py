from lpn_sample_generator.lpn_sample_generator import generate_sample
import numpy as np
import pytest
from random import randint

"""
Low effort unittesting for the generate_sample function. 
No monkeypatching even due to randomization since it would defeat the purpose of testing altogether.

TODO: Implement unittesting for the rest of the functionality.
"""


@pytest.mark.repeat(100)
@pytest.mark.parametrize(
    "args",
    [
        (
            np.random.randint(0, 2, randint(5,15)),
            0.1,
            randint(10,20),
        ),
        (
            np.random.randint(0, 2, randint(15,20)),
            0.125,
            randint(20,50),
        ),
        (
            np.random.randint(0, 2, 20),
            0.2,
            randint(50,100),
        ),
    ],
)
def test_generate_sample(args):
    A, b = generate_sample(*args)
    assert len(A) == args[2]
    assert len(A) == len(b)


@pytest.mark.parametrize(
    "args",
    [
        (
            np.random.randint(0, 2, randint(20,50)),
            0.2,
            100_000,
        ),
        (
            np.random.randint(0, 2, randint(50,100)),
            0.3,
            500_000,
        ),
        (
            np.random.randint(0, 2, randint(100,150)),
            0.325,
            1_000_000,
        ),
    ],
)
def test_generate_sample_stressful(args):
    A, b = generate_sample(*args)
    assert len(A) == args[2]
    assert len(A) == len(b)