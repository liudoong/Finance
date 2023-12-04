#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:05:11 2023

@author: DLIU
"""

import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_set(height, width, x_min, x_max, y_min, y_max, max_iter):
    """
    Gemerate an image of the Mandelbrot set

    Parameters
    ----------
    height : int
        Image height
    weidth : int
        Image weight
    x_min : float
        Minimum x-value
    x_max : float
        Maximum x-value
    y_min : float
        Minimum y-value
    y_max : float
        Maximum y-value
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    2D numpy array: A 2D array of integers indicating the iteration count at each point

    """
    
    x, y = np.linspace(x_min, x_max, width), np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C    = X + 1j ** Y + 0.4
    Z    = np.zeros_like(C)
    output = np.zeros(C.shape, dtype = int)
    
    for n in range(max_iter):
        mask    = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        output += mask
        
    return output

# parameters for the Mandelbrot set

height = 800
width  = 800
x_min, x_max = -2.5, 1.8
y_min, y_max = -2, 0.2
max_iter = 13

# Generate the Mandelbrot set

mandelbrot_image = mandelbrot_set(height, width, x_min, x_max, y_min, y_max, max_iter)

# plot the Mandelbrot set

plt.figure(figsize = (8, 15))
plt.imshow(mandelbrot_image, extent=[x_min, x_max, y_min, y_max], cmap = 'PuBuGn')
plt.axis('off')

#plt.colorbar()
#plt.title('Mandelbrot Set')
plt.show