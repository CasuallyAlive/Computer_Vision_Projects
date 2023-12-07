"""
This file holds the main code for disparity map calculations
"""
import torch
import numpy as np

from typing import Callable, Tuple


def calculate_disparity_map(left_img: torch.Tensor,
                            right_img: torch.Tensor,
                            block_size: int,
                            sim_measure_function: Callable,
                            max_search_bound: int = 50) -> torch.Tensor:
    """
    Calculate the disparity value at each pixel by searching a small 
    patch around a pixel from the left image in the right image

    Note: 
    1.  It is important for this project to follow the convention of search
        input in left image and search target in right image
    2.  While searching for disparity value for a patch, it may happen that there
        are multiple disparity values with the minimum value of the similarity
        measure. In that case we need to pick the smallest disparity value.
        Please check the numpy's argmin and pytorch's argmin carefully.
        Example:
        -- diparity_val -- | -- similarity error --
        -- 0               | 5 
        -- 1               | 4
        -- 2               | 7
        -- 3               | 4
        -- 4               | 12

        In this case we need the output to be 1 and not 3.
    3. The max_search_bound is defined from the patch center.

    Args:
    -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                C will be >= 1.
    -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
    -   block_size: the size of the block to be used for searching between
                    left and right image
    -   sim_measure_function: a function to measure similarity measure between
                            two tensors of the same shape; returns the error value
    -   max_search_bound: the maximum horizontal distance (in terms of pixels) 
                        to use for searching
    Returns:
    -   disparity_map: The map of disparity values at each pixel. 
                        Tensor of shape (H-2*(block_size//2),W-2*(block_size//2))
    """

    assert left_img.shape == right_img.shape
    disparity_map = torch.zeros(1) #placeholder, this is not the actual size
    ############################################################################
    # Student code begin
    ############################################################################
    
    H, W, C = left_img.shape; b_h = int(block_size//2)
    disparity_map = torch.zeros(size=(int(H-2*b_h), int(W-2*b_h)))
    
    def compare_patches(y, x, p1):
        x_min = max(b_h, x - max_search_bound)
        x_max = min(W - b_h, x + 1)
        
        min_error = np.Infinity; min_idx = (y, x)
        for xi in range(x_min, x_max):
            p2 = right_img[int(y-b_h):int(y+b_h)+1, (xi-b_h):(xi+b_h)+1, :]
            
            error = sim_measure_function(p1, p2)
            if error < min_error or (error == min_error and \
                                    np.abs(x - xi) < np.abs(x - min_idx[1])):
                min_idx = (y,xi)
                
            min_error = min(error, min_error)
                
        return min_idx
    
    min_x = b_h
    max_x = min([W-b_h, disparity_map.shape[1]])
    
    min_y = b_h
    max_y = min([H-b_h, disparity_map.shape[0]])
    
    # going over patches on left image and then calculating disparity values.
    for yl in range(min_y, max_y):
        for xl in range(min_x, max_x):
            
            p1 = left_img[int(yl-b_h):int(yl+b_h)+1, (xl-b_h):(xl+b_h)+1, :]

            _, x_min = compare_patches(yl, xl, p1)

            disparity_map[yl, xl] = abs(xl - x_min)
    
    ############################################################################
    # Student code end
    ############################################################################
    return disparity_map

def calculate_cost_volume(left_img: torch.Tensor,
                          right_img: torch.Tensor,
                          max_disparity: int,
                          sim_measure_function: Callable,
                          block_size: int = 9):
    """
    Calculate the cost volume. Each pixel will have D=max_disparity cost values
    associated with it. Basically for each pixel, we compute the cost of
    different disparities and put them all into a tensor.

    Note: 
    1.  It is important for this project to follow the convention of search
        input in left image and search target in right image
    2.  If the shifted patch in the right image will go out of bounds, it is
        good to set the default cost for that pixel and disparity to be something
        high(we recommend 255), so that when we consider costs, valid disparities will have a lower
        cost. 

    Args:
    -   left_img: image from the left stereo camera. Torch tensor of shape (H,W,C).
                C will be 1 or 3.
    -   right_img: image from the right stereo camera. Torch tensor of shape (H,W,C)
    -   max_disparity:  represents the number of disparity values we will consider.
                    0 to max_disparity-1
    -   sim_measure_function: a function to measure similarity measure between
                    two tensors of the same shape; returns the error value
    -   block_size: the size of the block to be used for searching between
                    left and right image
    Returns:
    -   cost_volume: The cost volume tensor of shape (H,W,D). H,W are image
                dimensions, and D is max_disparity. cost_volume[x,y,d] 
                represents the similarity or cost between a patch around left[x,y]  
                and a patch shifted by disparity d in the right image. 
    """
    #placeholder
    H = left_img.shape[0]
    W = right_img.shape[1]
    cost_volume = torch.zeros(H, W, max_disparity)
    ############################################################################
    # Student code begin
    ############################################################################
    b_h = int(block_size//2)
    def compare_patches(y, x, p1):
        x_min = max(b_h, x - max_disparity+1)
        x_max = min(W - b_h, x_min+max_disparity)
        errors = torch.zeros(size=(max_disparity,))
        for xi in range(x_min, x_max):
            p2 = right_img[int(y-b_h):int(y+b_h)+1, (xi-b_h):(xi+b_h)+1, :]

            errors[abs(x - xi)] = sim_measure_function(p1, p2)
                
        return errors

    min_x = b_h
    max_x = min([W-b_h, cost_volume.shape[1]])

    min_y = b_h
    max_y = min([H-b_h, cost_volume.shape[0]])

    # going over patches on left image and then calculating disparity values.
    for yl in range(min_y, max_y):
        for xl in range(min_x, max_x):
            
            p1 = left_img[int(yl-b_h):int(yl+b_h)+1, (xl-b_h):(xl+b_h)+1, :]

            cost_volume[yl, xl,:] = compare_patches(yl, xl, p1)

  ############################################################################
  # Student code end
  ############################################################################
    return cost_volume
