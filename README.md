# Faster DPFN in C++

This reposity contains the C++ implementation of DPFN, which was published in the AAAI 2024 conference, under the title:
'Protect Your Score: Contact Tracing With Differential Privacy Guarantees'

The original Python-based implementation is at [github.com/RobRomijnders/dpfn_aaai](https://github.com/RobRomijnders/dpfn_aaai).

This implementation is about two times faster and more memory efficient. The original library could not run population
sizes larger than 500.000 users due to memory constraints. This implementation can run 16 parallel simulations with DPFN on a simulator of 100.000 agents in about thirty minutes. Please refer to the original repository for more information on the accompanying paper and academic references.

This repository is an adaptation of [this pybind template.](https://github.com/pybind/python_example)

## Important data structures

Central to the implementation of FN and BP is the array 'past_contacts.' As both algorithms are run on irregular graphs, we need a way to quickly index the contacts of each agent when doing decentralized inference.
To this end, 'past_contacts' maintains a list, per agent, of all the incoming contacts. Then during the decentralized inference step, the local agent need only to run over this list, which contains the timestep
of the contact and a sender id. For sake of simplicity, this array is padded with -1 to a fixed size length. This fixed-size length is a global parameter named 'CTC_SIZE.' Given that an agent has on average 14 contacts per day,
the array will be (900-144)/900=84% sparse.

Specifically for FN, the inference is sped up by running inference over possible sequences. Instead of considering O(4^T) possible sequences of S,E,I,R, we run inference only over the possible sequences S->E->I->R, which are only O(T^3).
Sequences are then scaled by their log likelihood. You may encounter the different terms of the log likelihoods. These correspond to: a_terms, which are the prior transition probabilities, c_terms, which are due to observations, and d_terms, which are due to contacts whose infectiousness can influence transitions.

## Installation

For installation, one can use `make build` from the main directory. Prior to any experiment, it is recommended to run the tests with `make test`.
Some of the unit tests compare this library against the [original library](https://github.com/RobRomijnders/dpfn_aaai).

## Questions and communication
Any correspondence may go to romijndersrob@gmail.com or r.romijnders@uva.nl