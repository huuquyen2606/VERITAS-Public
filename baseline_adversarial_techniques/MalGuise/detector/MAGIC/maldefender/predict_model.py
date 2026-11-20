#!/usr/bin/python3.7
import sys
import os
import torch
import math
import random
import time
import glog as log
import numpy as np
from typing import Dict, List
from ml_utils import cmd_args, gHP, S2VGraph
from ml_utils import loadGraphsMayCache, normalizeFeatures
from ml_utils import filterOutNoEdgeGraphs
from ml_utils import loadModel, saveModel, decideHyperparameters
from e2e_model import Classifier, predictDataset


def exportPredictions(graphs: List[S2VGraph], predProb):
    """Export binary classification results: Id, MalwareProb (probability [0,1])"""
    log.debug(f'Export {len(predProb)} predictions for {len(graphs)} graphs')
    assert len(graphs) == len(predProb)

    output = open(cmd_args.test_dir + '/submission.csv', 'w')
    output.write('Id,MalwareProb\n')
    for (i, g) in enumerate(graphs):
        assert len(predProb[i]) == gHP['numClasses']
        # predProb[i][1] = probability of class 1 (malware)
        malware_prob = predProb[i][1] if gHP['numClasses'] == 2 else max(predProb[i][1:])
        output.write('%s,%.8f\n' % (g.bId, malware_prob))

    output.close()
    log.info(f'Predictions saved to {cmd_args.test_dir}/submission.csv')


def testWithModel(testGraphs: List[S2VGraph]) -> None:
    classifier = Classifier()
    loadModel(classifier)
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    classifier.eval()
    startTime = time.process_time()
    testPredProb = predictDataset(testGraphs, classifier)
    log.info(f'Net test time = {time.process_time() - startTime} seconds')
    exportPredictions(testGraphs, testPredProb)


if __name__ == '__main__':
    log.setLevel("INFO")
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    if cmd_args.data == 'YANACFG':
        log.warning(f'No testset for YANACFG data')
    else:
        startTime = time.process_time()
        testGraphs = loadGraphsMayCache(cmd_args.test_dir, isTestSet=True)
        normalizeFeatures(testGraphs, isTestSet=True,
                          operation=cmd_args.norm_op)
        dataReadyTime = time.process_time() - startTime
        log.info('Testset ready takes %.2fs' % dataReadyTime)

        # Load poolingK from cache to avoid parsing train set
        import pickle
        try:
            with open(cmd_args.cache_path + '.pkl', 'rb') as f:
                cached_data = pickle.load(f)
                
                # Get correct properties
                gHP['numClasses'] = cached_data.get('numClasses', 2)
                gHP['featureDim'] = cached_data.get('featureDim', 33)
                
                # Get the filtered graphs to compute the exact poolingK
                from ml_utils import filterOutNoEdgeGraphs
                train_graphs = filterOutNoEdgeGraphs(cached_data['graphs'])
                
                import math
                numNodesList = sorted([g.num_nodes for g in train_graphs])
                
                from hyperparameters import parseHpTuning, HyperParameterIterator
                if cmd_args.hp_path != 'none':
                    hpIter = HyperParameterIterator(cmd_args.hp_path)
                    hp = next(hpIter)
                    poolingRatio = hp['poolingRatio']
                else:
                    poolingRatio = 0.64
                
                idx = int(math.ceil(poolingRatio * len(train_graphs))) - 1
                gHP['poolingK'] = numNodesList[idx]
        except Exception as e:
            log.warning(f"Could not load poolingK from cache: {e}")
            gHP['poolingK'] = 157
            gHP['numClasses'] = 2
        class DummyGraph:
            def __init__(self, n):
                self.num_nodes = n
        decideHyperparameters([DummyGraph(gHP['poolingK'])])
        testWithModel(testGraphs)
