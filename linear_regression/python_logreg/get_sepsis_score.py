#!/usr/bin/env python

import numpy as np

def get_sepsis_score(data, model):
    x_mean = np.array([ 
   22.99,  85.11,  97.13,  36.86, 122.36,  81.77,  63.23,  18.94,  -0.46,   0.49,
   7.39,  40.63,  23.76,   7.82,   1.59, 130.87,   2.05,   4.15,  31.17,  10.38,
  11.08, 196.6,   62.77])
    x_std = np.array([ 
   16.86,  16.74,   3.04,   0.72,  23.14,  16.32,  14.02,   5.17,   3.83,   0.19,
   0.06,   8.45,  19.63,  2.1,    2.02,  47.45,   0.38,   0.61,   5.65,   1.95,
   5.87, 100.95,  15.97])

    for i in range(1, len(data)):
        mask = np.isnan(data[i])
        data[i][mask] = data[i - 1][mask]

    x = data[-1, 0:23]
    x_norm = np.nan_to_num((x - x_mean) / x_std)

    coeffs= np.array([
   0.1871,  0.2689, -0.0355,  0.1505,  0.0301, -0.2952,  0.1208,  0.0678,  0.183,
   0.1742, -0.0382,  0.0088,  0.1823, -0.1392,  0.1094,  0.0315, -0.0465, -0.0341,
  -0.1274,  0.0514,  0.1089,  0.009,   0.039])
    const = -3.9766

    z = const + np.dot(x_norm, coeffs)
    score = 1.0 / (1 + np.exp(-z))
    score = min(max(score,0),1)
    label = score > 0.0262

    return score, label

def load_sepsis_model():
    return None
