#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 00:40:51 2025

@author: october
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint

# -----------------------------
# Section 1: Core Functions
# -----------------------------

def cumulative_hazard(hazard_curve, dt):
    """Compute cumulative hazard from hazard curve."""
    return np.sum(hazard_curve) * dt

def cumulative_PD(cum_hazard):
    """Compute cumulative default probability from cumulative hazard."""
    return 1 - np.exp(-cum_hazard)

def portfolio_PD(PDs, weights):
    """Compute portfolio-level PD as weighted average."""
    return np.sum(PDs * weights) / np.sum(weights)

def conservative_indicator(D):
    """Indicator: 1 if constant hazard PD >= term-structured PD, else 0."""
    return (D >= 0).astype(int)

def clopper_pearson_interval(I, alpha=0.05):
    """Compute one-sided Clopper-Pearson confidence interval for proportion."""
    n = len(I)
    k = np.sum(I)
    lower, upper = proportion_confint(k, n, alpha=alpha, method='beta')
    return lower, upper