# CUDA Quantum State Ordering

CUDA Quantum uses the Least Significant Bit (LSB) ordering schema for qubit
registers, qubit states, and system state vectors. This can be seen with the
following example. Given a register of 5 qubits, all initialized to the 0-state,
the qubit indices are

```text
-|---------|---------|---------|---------|
[0]       [1]       [2]       [3]       [4]
```

and its state is

```text
|0>|0>|0>|0>|0> = |00000>
collapses to: 00000
```

If we flip the 0th qubit to the 1-state, the state according to the CUDA Quantum
schema is now

```text
|1>|0>|0>|0>|0> = |10000>
collapses to: 10000
```

## Alternative bit ordering schemes

This is opposed to a Most Significant Bit (MSB) scheme, often seen in simulators
such as Qiskit Aer. Using the same example as above, the qubit indices become

```text
-|---------|---------|---------|---------|
[4]       [3]       [2]       [1]       [0]
```

with the state

```text
|0>|0>|0>|0>|0> = |00000>
collapses to: 00000
```

If we flip the 0th qubit to the 1-state, the state according to the CUDA Quantum
schema is now

```text
|0>|0>|0>|0>|1> = |00001>
collapses to: 00001
```

__Users of CUDA Quantum should *always* index qubits according to the LSB
convention.__

## Notes of Caution

While MSB is the typical convention chosen in textbooks, it's not uncommon to
find different ordering schemes across the literature. Given the impact of the
different orderings on the gate matrices, here are explicit definitions for a
small subset of the gate matrices in CUDA Quantum.

```text
X = ((0,1),
    (1,0))

Y = ((0,-i),
    (i,0))

CS = ((1,0,0,0),
      (0,1,0,0),
      (0,0,1,0),
      (0,0,0,i))

CH = ((1,0,0,0),
      (0,1,0,0),
      (0,0,1/sqrt(2),1/sqrt(2)),
      (0,0,1/sqrt(2),-1/sqrt(2)))
```

Additionally, qiskit will often remove certain multiplicative factors from their
parameterized gate definitions. To avoid confusion, here is how we define the
different parameterized gates in CUDA Quantum. Note: where we call, e.g,
`cos(theta/2)`, qiskit often uses `cos(theta)`. As with the gates above, the
following matrices are ordered for MSB.

```text
RX(theta) = ((cos(theta/2), -isin(theta/2)),
             (-isin(theta/2), cos(theta/2)))

RY(theta) = ((cos(theta/2), -sin(theta/2)),
             (sin(theta/2), cos(theta/2)))

RZ(lambda) = ((exp(-i*lambda/2), 0),
              (0, exp(i*lambda/2)))

CPHASE(theta) = ((1,0,0,0),
                  (0,1,0,0),
                  (0,0,1,0),
                  (0,0,0,exp(i*theta))

U1(phi) = ((1,0),
           (0,exp(i*phi)))

U = U3(theta,phi,lambda) = ((
                            cos(theta/2),
                            exp(i*phi) * sin(theta/2)
                          ),
                          (
                            -1.*exp(i*lambda) * sin(theta/2),
                            exp(i*(phi+lambda)) * cos(theta/2)
                          ))


```
