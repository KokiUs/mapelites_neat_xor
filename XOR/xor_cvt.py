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

import math
import numpy as np
import multiprocessing
# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree
import xor_common as cm
import xor_mapelites as me
import neat
import visualize 
import os
from neat.six_util import iteritems, itervalues

def __add_to_archive(s, centroid, archive, kdt):            #アーカイブに追加
    niche_index = kdt.query([centroid], k=1)[1][0][0]       #niche_indexに、対応するポイントのネイバーのインデックスのリスト[1][0][0]を代入
    niche = kdt.data[niche_index]                           #niche_indexからniche
    n = cm.make_hashable(niche)                             #nicheをfloat型に変換
    s.centroid = n                                          #sはnに含まれる
    if n in archive:                                        #もしarchive内にsと同じn（ニッチ）があれば
        if s.fitness > archive[n].fitness:                  #fitnessを比較し、sの方が大きければsに入れ替わる
            archive[n] = s
            return 1
        return 0
    else:
        archive[n] = s                                      #もしarchive内にsと同じn（ニッチ）がなければ単に入る
        return 1


# evaluate a single vector (x) with a function f and return a species
def __evaluate(t,config):          #遺伝子型から適応度と種を評価
    s = [0]*100
    g = me.eval_genomes(t,config)                                                #個体の適応度計算
    s = [cm.Species(ge[1], [ge[1].connections[(-1, 0)].weight,ge[1].connections[(-2, 0)].weight], 
    ge[1].fitness) for ge in g]
    
    return s

# map-elites algorithm (CVT variant)
def compute(dim_map, dim_x, f,
            n_niches=1000,
            max_evals=1e3,
            params=cm.default_params,
            log_file=None,
            variation_operator=cm.variation):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.
       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile
    """
    # setup the parallel processing pool（並列処理プールを設定）
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # create the CVT（重心ボロノイ分割を生成）
    c = cm.cvt(n_niches, dim_map,
              params['cvt_samples'], params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    cm.__write_centroids(c)

    archive = {} # init archive (empty)
    n_evals = 0 # number of evaluations since the beginning
    b_evals = 0 # number evaluation since the last dump

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    p = neat.Population(config)

    # main loop
    while (n_evals < max_evals):    #最大評価回数まで以下を実行
        #to_evaluate = []
        to_evaluate = {}
        # random initialization（ランダムな初期化．ここでは100個体）
        if len(archive) <= params['random_init'] * n_niches:    #'random_init': 0.1
            to_evaluate = (p.reproduction.create_new(config.genome_type, config.genome_config,params['random_init_batch']))

        else:  # variation/selection loop（突然変異のみ）
            keys = list(archive.keys())                         #アーカイブはニッチarchive.keys()とその適応度を持つ．
            # we select all the parents at the same time because randint is slow
            rand1 = np.random.randint(len(keys), size=params['batch_size'])     #アーカイブの個数以下の数値の一様乱数をbatch_size個（100）生成
            #rand2 = np.random.randint(len(keys), size=params['batch_size'])
            
            for n in range(0, params['batch_size']):                            #batch_size個体（100個体）の子孫を生成
                # parent selection（親を選択）
                x = archive[keys[rand1[n]]]
                #y = archive[keys[rand2[n]]]
                
                # copy & add variation（複製と交叉）
                #z = variation_operator(x.x, y.x, params)
                
                #突然変異のみ
                z = p.reproduction.reproduce3(x.x, config)

                to_evaluate.update(z)                                        #生成された集団を関数fで評価する用

        # evaluation of the fitness for to_evaluate（to_evaluateに入った個体の適応度を評価）
        #s_list = cm.parallel_eval(__evaluate, to_evaluate, pool, params)    #個体を並列評価

        s_list = cm.serial_eval(__evaluate, list(iteritems(to_evaluate)), pool, params,config)

        # natural selection（自然選択）
        for s in s_list:                #評価された個体をアーカイブに追加
            __add_to_archive(s, s.desc, archive, kdt)
        # count evals
        n_evals += len(to_evaluate)     #評価回数（個体数）をカウント
        b_evals += len(to_evaluate)
        print("n_evals/max_evals"+str(n_evals)+"/"+str(max_evals))

        # write archive（アーカイブを書き出す）
        if b_evals >= params['dump_period'] and params['dump_period'] != -1:
            #print("[{}/{}]".format(n_evals, int(max_evals)), end=" ", flush=True)
            cm.__save_archive(archive, n_evals)
            b_evals = 0
        # write log（ログを書き出す）
        if log_file != None:
            fit_list = np.array([x.fitness for x in archive.values()])
            log_file.write("{} {} {} {} {} {} {}\n".format(n_evals, len(archive.keys()),
                    fit_list.max(), np.mean(fit_list), np.median(fit_list),
                    np.percentile(fit_list, 5), np.percentile(fit_list, 95)))
            log_file.flush()
    cm.__save_archive(archive, n_evals)

    #ネットワークを描画
    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    #visualize.draw_net(config, list(archive.values())[0].x, True, node_names=node_names)
    #visualize.draw_net(config, list(archive.values())[1].x, True, node_names=node_names)
    #visualize.draw_net(config, list(archive.values())[2].x, True, node_names=node_names)
    
    return archive
