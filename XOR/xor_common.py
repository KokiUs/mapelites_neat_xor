#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
#

import math
import numpy as np
import multiprocessing
from pathlib import Path
import sys
import random
from collections import defaultdict
from sklearn.cluster import KMeans
from neat.six_util import iteritems, itervalues

#パラメータ設定
default_params = \
    {
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to paralleliez
        #"batch_size": 100,
        "batch_size": 50,
        # proportion of niches to be filled before starting
        "random_init": 0.1,
        # batch for random initialization
        #"random_init_batch": 100,
        "random_init_batch": 50,
        # when to write results (one generation = one batch)
        "dump_period": 10000,
        # do we use several cores?
        "parallel": False,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": True,
        # min/max of parameters
        "min": 0,
        "max": 1,
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2
    }

class Species:     #種
    def __init__(self, x, desc, fitness, centroid=None):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid


def polynomial_mutation(x):     #変異
    '''
    Cf Deb 2001, p 124 ; param: eta_m
    '''
    y = x.copy()
    eta_m = 5.0;
    r = np.random.random(size=len(x))
    for i in range(0, len(x)):
        if r[i] < 0.5:
            delta_i = math.pow(2.0 * r[i], 1.0 / (eta_m + 1.0)) - 1.0
        else:
            delta_i = 1 - math.pow(2.0 * (1.0 - r[i]), 1.0 / (eta_m + 1.0))
        y[i] += delta_i
    return y

def sbx(x, y, params):      #xとyの交叉（eta高：近い親同士、eta低：遠い親同士）
    '''
    SBX (cf Deb 2001, p 113) Simulated Binary Crossover

    A large value ef eta gives a higher probablitity for
    creating a `near-parent' solutions and a small value allows
    distant solutions to be selected as offspring.
    '''
    eta = 10.0
    xl = params['min']  #0
    xu = params['max']  #1
    z = x.copy()
    r1 = np.random.random(size=len(x))
    r2 = np.random.random(size=len(x))

    for i in range(0, len(x)):
        #xとyを比べ，差が1e-15より大きい要素に対して以下の計算結果を代入．差が1e-15より小さければxの値を引き継ぐ
        if abs(x[i] - y[i]) > 1e-15:
            x1 = min(x[i], y[i])    #小さい方
            x2 = max(x[i], y[i])    #大きい方

            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            rand = r1[i]
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            c1 = min(max(c1, xl), xu)
            c2 = min(max(c2, xl), xu)

            if r2[i] <= 0.5:
                z[i] = c2
            else:
                z[i] = c1
    return z        #子zを返す


def iso_dd(x, y, params):       #交叉
    '''
    Iso+Line
    Ref:
    Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
    GECCO 2018
    '''
    assert(x.shape == y.shape)      #x.shape == y.shapeがTrueかどうかテスト．Falseならば例外を投げる
    p_max = np.array(params["max"])
    p_min = np.array(params["min"])
    a = np.random.normal(0, params['iso_sigma'], size=len(x))
    b = np.random.normal(0, params['line_sigma'])
    norm = np.linalg.norm(x - y)
    z = x.copy() + a + b * (x - y)
    return np.clip(z, p_min, p_max)


def variation(x, z, params):    #sbxを実行
    assert(x.shape == z.shape)
    y = sbx(x, z, params)
    return y

def __centroids_filename(k, dim):     #centroidsの保存用のファイル名を作る
    return 'centroids_' + str(k) + '_' + str(dim) + '.dat'


def __write_centroids(centroids):     #centroidsの値をファイルに保存
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')


def cvt(k, dim, samples, cvt_use_cache=True):   #一様分布のcentroidsの値を計算
    # check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)

    x = np.random.rand(samples, dim)    #サイズ(samples, dim)の一様分布（0.0以上、1.0未満）xを生成
    """k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, n_jobs=-1, verbose=1)#,algorithm="full")"""
    k_means = KMeans(init='k-means++', n_clusters=k,            #K-Means 法 (K-平均法）によるクラスタ分析
                     n_init=1, verbose=1)#,algorithm="full")
    k_means.fit(x)                                              #xにK-Means 法を適用
    __write_centroids(k_means.cluster_centers_)                 #centroidsの値をファイルに保存

    return k_means.cluster_centers_         #centroidsの値を返す


def make_hashable(array):
    return tuple(map(float, array))     #arrayをfloat型にしたものをタプルで返す


def parallel_eval(evaluate_function, to_evaluate, pool, params):    #並列に評価
    if params['parallel'] == True:
        s_list = pool.map(evaluate_function, to_evaluate)       #マルチプロセス処理（to_evaluateの要素を並列にevaluate_functionにかける）
    else:
        print(to_evaluate)
        s_list = map(evaluate_function, to_evaluate)            #直列に処理
    return list(s_list)                                         #個体とその評価（fitness, desc）を返す


def serial_eval(evaluate_function, to_evaluate, pool, params,config):    #自作。一つの評価関数の実行のみ
    s_list = evaluate_function(to_evaluate,config)
    #print(len(s_list))
    return list(s_list)


# format: fitness, centroid, desc, genome \n
# fitness, centroid, desc and x are vectors
def __save_archive(archive, gen):                       #アーカイブを保存
    """def write_array(a, f):
        for i, s in list(iteritems(a)):
            f.write(str(s) + ' ')"""
    filename = 'archive_' + str(gen) + '.dat'
    with open(filename, 'w') as f:
        for k in archive.values():
            f.write(str(k.fitness) + ' ')
            #write_array(k.centroid, f)
            #write_array(k.desc, f)
            #write_array(k.x, f)
            f.write(str(k.centroid[0]) + ' ')
            f.write(str(k.centroid[1]) + ' ')
            f.write(str(k.desc[0]) + ' ')
            f.write(str(k.desc[1]) + ' ')
            #f.write(str(k.x) + ' ')
            f.write("\n")