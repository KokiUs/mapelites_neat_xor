"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import os

#from neat import __init__
import neat
import visualize

import xor_common
import xor_cvt

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):     #適応度関数
    #print("genomes"+str(genomes))
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2

    return genomes

def train():
    px = xor_common.default_params.copy()
    archive = xor_cvt.compute(2, 4, eval_genomes, n_niches=10000, max_evals=1e5, log_file=open('xor.dat', 'w'), params=px)

if __name__ == '__main__':
    train()
