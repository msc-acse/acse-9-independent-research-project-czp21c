#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:30:03 2019

@author: Zongpeng Chen
@Github Alias: czp21c
"""

import numpy as np
import torch
import torch.nn.functional as F
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    device = 'cuda'


def separate_train_test(fig, fig_c, ratio=0, select=[]):
    """
       Return the separated training and test datasets.

       fig - a numpy array, the original seismic sections.
       fig_c - a numpy array, the class label map of the original seismic sections.
       ratio - a number between 0 to 1, the ratio of whole seismic sections are selected to be training dataset.
       select - a list, selected seismic sections are selected to be training dataset.
    """
    lend = len(fig)  # the total number of sections.

    # select training sections as a ratio to the total sections.
    if ratio != 0:
        ref = int(ratio * lend)
    
        fig_t = []
        fig_tc = []
    
        while len(fig) > ref:
            num = np.random.randint(0, len(fig))
            fig_t.append(fig[num])
            fig_tc.append(fig_c[num])
            fig = np.delete(fig, num, 0)
            fig_c = np.delete(fig_c, num, 0)

    # select specific sections as training sections.
    else:
        fig_new = []
        figc_new = []
        fig_t = []
        fig_tc = []
        for i in range(len(fig)):
            if i in select:
                fig_new.append(fig[i])
                figc_new.append(fig_c[i])
            else:
                fig_t.append(fig[i])
                fig_tc.append(fig_c[i])
        fig = np.array(fig_new)
        fig_c = np.array(figc_new)
    
    fig_t = np.array(fig_t)
    fig_tc = np.array(fig_tc)
    return fig, fig_c, fig_t, fig_tc


def pad_data_zero(fig, minib_size):
    """
       Return the padded seismic sections.
       Pad the seismic sections with zero near the edge.
       The padding size is equal to the minib_size.

       fig - a numpy array, the original seismic sections.
       minib_size - an integer, size of the mini patch of each training object.
    """
    pad_size = minib_size//2  # determine the padding size.

    xconcatb = []
    xconcatm = []

    # create the padding chunks.
    for i in range(pad_size):
        xconcatbtemp = []
        for j in range(2*pad_size + fig.shape[1]):
            xconcatbtemp.append(0.)
        xconcatb.append(xconcatbtemp)    
    
    for i in range(len(fig)):
        xconcatmtemp = []
        for j in range(pad_size):
            xconcatmtemp.append(0.)
        xconcatm.append(xconcatmtemp)
    xconcatm = np.array(xconcatm)

    # concatenate the chunks to the original section.
    fig = np.concatenate((xconcatm, fig), axis=1)
    fig = np.concatenate((fig, xconcatm), axis=1)
    
    fig = np.concatenate((xconcatb, fig), axis=0)
    fig = np.concatenate((fig, xconcatb), axis=0)
    
    return np.array(fig)


def pad_data_mirror(fig, minib_size):
    """
       Return the padded seismic sections.
       Pad the seismic sections with mirroring data near the edge.
       The padding size is equal to the minib_size.

       fig - a numpy array, the original seismic sections.
       minib_size - an integer, size of the mini patch of each training object.
    """
    pad_size = minib_size//2  # determine the padding size.

    xconcatt = []
    xconcatb = []
    xconcatl = []
    xconcatr = []

    # create the padding chunks.
    for i in range(len(fig)):
        xconcatltemp = []
        xconcatrtemp = []
        for j in range(pad_size):
            xconcatltemp.append(fig[i][pad_size - j])
            xconcatrtemp.append(fig[i][-2 - j])
        xconcatl.append(xconcatltemp)
        xconcatr.append(xconcatrtemp)
    xconcatl = np.array(xconcatl)
    xconcatr = np.array(xconcatr)

    # concatenate the chunks to the original section.
    fig = np.concatenate((xconcatl, fig), axis=1)
    fig = np.concatenate((fig, xconcatr), axis=1)

    # create the padding chunks.
    for i in range(pad_size):
        xconcatttemp = []
        xconcatbtemp = []
        for j in range(fig.shape[1]):
            xconcatttemp.append(fig[pad_size - i][j])
            xconcatbtemp.append(fig[-2 - i][j])
        xconcatt.append(xconcatttemp)
        xconcatb.append(xconcatbtemp) 
    xconcatt = np.array(xconcatt)
    xconcatb = np.array(xconcatb)

    # concatenate the chunks to the original section.
    fig = np.concatenate((xconcatt, fig), axis=0)
    fig = np.concatenate((fig, xconcatb), axis=0)
    
    return np.array(fig)


def get_effective_data(sep_indices, ratio, char_c1, char_c7):
    """
       Return effective training coordinates datasets by balancing the dataset and adding structural information.

       sep_indices - a list, sorted coordinates by class label.
       ratio - a number, the ratio of the number of the class having the least training objects will be set
         as the benchmark.
       char_c1 - a list, the characteristic objects of the first class.
       char_c7 - a list, the characteristic objects of the last class.
    """

    num = len(sep_indices[0])  # get the number of classes.

    # get the benchmark.
    for i in range(len(sep_indices)):
        if len(sep_indices[i]) <= num:
            num = len(sep_indices[i])  

    # implement the balancing.
    for i in range(len(sep_indices)):
        if i == 0:
            n = np.random.randint(0.98 * ratio * num, 1.02 * ratio * num)
            while len(sep_indices[i]) < n - len(char_c1):
                sep_indices[i].append(sep_indices[i][np.random.randint(0, len(sep_indices[i]))])
        
            while len(sep_indices[i]) > n - len(char_c1): 
                del(sep_indices[i][np.random.randint(0, len(sep_indices[i]))]) 
            
            for indice in char_c1:
                sep_indices[i].append(indice)
                
        elif i == len(sep_indices) - 1:
            n = np.random.randint(0.98 * ratio * num, 1.02 * ratio * num)
            while len(sep_indices[i]) < n - len(char_c7):
                sep_indices[i].append(sep_indices[i][np.random.randint(0, len(sep_indices[i]))])
        
            while len(sep_indices[i]) > n - len(char_c7): 
                del(sep_indices[i][np.random.randint(0, len(sep_indices[i]))]) 
            
            for indice in char_c7:
                sep_indices[i].append(indice)

        else:
            n = np.random.randint(0.98 * ratio * num, 1.02 * ratio * num)
            while len(sep_indices[i]) < n:
                sep_indices[i].append(sep_indices[i][np.random.randint(0, len(sep_indices[i]))])
        
            while len(sep_indices[i]) > n: 
                del(sep_indices[i][np.random.randint(0, len(sep_indices[i]))])  

    # get the effective datasets.
    final_indices = []
    final_labels = []
    
    for i in range(len(sep_indices)):
        for j in range(len(sep_indices[i])):
            final_indices.append(sep_indices[i][j])
            final_labels.append(i)
            
    return np.array(final_indices), np.array(final_labels)


def save_coord(fig, fig_c, minib_size):
    """
        Return the validation/test coordinates datasets.

        fig - a numpy array, the original seismic sections.
        fig_c - a numpy array, the class label map of the original seismic sections.
        minib_size - an integer, size of the mini patch of each training object.
    """
    X_val = []
    y_val = []
    if len(fig.shape) == 3:
        for i in range(fig.shape[0]):
            for j in range(minib_size, fig.shape[1] - minib_size):
                for k in range(minib_size, fig.shape[2] - minib_size):
                    X_val.append([i, j, k])
                    y_val.append(fig_c[i, j-minib_size, k-minib_size]-1)
    elif len(fig.shape) == 4:
        for i in range(fig.shape[0]):
            for j in range(minib_size, fig.shape[2] - minib_size):
                for k in range(minib_size, fig.shape[3] - minib_size):
                    X_val.append([i, j, k])
                    y_val.append(fig_c[i, j-minib_size, k-minib_size]-1)
    
    return np.array(X_val), np.array(y_val)


def find_interested_area(fig, fig_c):
    """
        Return two coordinates.
        Find the positions where the seismic section begins and stops introducing pixels in different classes.

        fig - a numpy array, the original seismic sections.
        fig_c - a numpy array, the class label map of the original seismic sections.
    """
    cr_xs = 0
    cr_ys = 0
    cr_xe = fig_c.shape[1]
    cr_ye = fig_c.shape[2]
    for i in range(len(fig_c)):
        interest_indxt, interest_indexyt = np.where(fig_c[i] != 1)
        interest_indxb, interest_indexyb = np.where(fig_c[i] != 7)
        interest_indxt, interest_indexyt = interest_indxt[0], interest_indexyt[0]
        interest_indxb, interest_indexyb = interest_indxb[-1], interest_indexyb[-1]
        if interest_indxt > cr_xs:
            cr_xs = interest_indxt
        if interest_indexyt > cr_ys:
            cr_ys = interest_indexyt
        if interest_indxb < cr_xe:
            cr_xe = interest_indxb
        if interest_indexyb < cr_ye:
            cr_ye = interest_indexyb
    cr_xs = max(0, cr_xs - 50)
    cr_xe = min(fig_c.shape[1], cr_xe + 50)
    return cr_xs, cr_xe


def crop_interest_area(fig, fig_c, num):
    """
        Return the coordinates at where to crop the sections
        and return the cropped seismic sections and class label maps.
        Design for encoder-decoder CNN model.

        fig - a numpy array, the original seismic sections.
        fig_c - a numpy array, the class label map of the original seismic sections.
        num - an interger, the number needs to be divided exactly.
    """
    interest_range_v = fig_c.shape[1]
    interest_range_h = fig_c.shape[2]
    ref = interest_range_v % num
    ref_t = ref // 2
    ref_b = ref - ref_t

    # get the crop indices.
    interest_t = ref_t
    interest_b = interest_range_v - ref_b

    ref_y = interest_range_h % num
    ref_l = ref_y // 2
    ref_r = ref_y - ref_l

    # get the crop indices.
    interest_l = ref_l
    interest_r = interest_range_h - ref_r

    # crop the sections for '6-feature' dataset.
    if len(fig.shape) == 4:
        fig = fig[:, :, interest_t:interest_b, interest_l:interest_r]

    # crop the sections for '1-feature' dataset.
    if len(fig.shape) == 3:
        fig = fig[:, interest_t:interest_b, interest_l:interest_r]

    # crop the sections for class label map.
    fig_c = fig_c[:, interest_t:interest_b, interest_l:interest_r]

    return interest_t, interest_b, interest_l, interest_r, fig, fig_c


def ff_data_augmentation(patch, ref):
    """
        Return a new training data after data augmentation.
        Design for feed-forward DNN model.

        patch - a numpy array, the original training object.
        ref - a list, a moduler to the augmented data.
    """
    patch = patch.clone()
    temp = (torch.rand(len(ref)) + torch.rand(len(ref)) * (-1)).numpy()
    patch = torch.FloatTensor(patch.numpy() + ref * temp)
    return patch


def ff_dynamic_data_augmentation(patch, ref, prob):
    """
        Return a new training data after dynamic data augmentation.
        Design for feed-forward DNN model.

        patch - a numpy array, the original training object.
        ref - a list, a moduler to the augmented data.
        prob - a number, between 0 to 1, the probability of the original training object to be transformed.
    """
    patch = patch.clone()
    for i in range(len(patch)):
        if np.random.uniform() < prob:
            patch[i] += ref[i] * (np.random.uniform() + np.random.uniform() * (-1))
    return patch


def data_augmentation(patch, minb, maxb, p_transform):
    """
        Return a new training data after data augmentation.
        Design for mini-patch CNN model.

        patch - a numpy array, the original training object.
        minb - a number, the minimum of whole seismic section.
        maxb - a number, the maximum of whole seismic section.
        p_transform - a torchvision.tranform function, the type of transform.
    """
    patch = patch.clone()
    # for '1-feature' dataset.
    if len(patch.shape) == 2:
        patch = ((255.*(patch-minb))/(maxb-minb)).int()
        img = (p_transform(patch)[0]).float()  # apply the transformation.
        s1 = patch.shape[0]
        s2 = patch.shape[1]
        temp = torch.rand(s1, s2) + torch.rand(s1, s2)*(-1)

        # avoid unexpected bugs from PyTorch
        if torch.max(img) > 100:
            patch = img + temp
        else:
            patch = (255 * img) + temp
        patch = ((patch * (maxb-minb)) / 255.) + minb

    # for '6-feature' dataset.
    if len(patch.shape) == 3:           
        for i in range(patch.shape[0]):          
            patch[i] = ((255.*(patch[i]-minb[i]))/(maxb[i]-minb[i])).int()
            img = (p_transform(patch[i])[0]).float()  # apply the transformation.
            s1 = patch[i].shape[0]
            s2 = patch[i].shape[1]
            temp = torch.rand(s1, s2) + torch.rand(s1, s2)*(-1)

            # avoid unexpected bugs from PyTorch
            if torch.max(img) > 100:
                patch[i] = img + temp
            else:
                patch[i] = (255 * img) + temp
            patch[i] = ((patch[i] * (maxb[i]-minb[i])) / 255.) + minb[i]
    return patch


def dynamic_data_augmentation(patch, minb, maxb, d_transform, prob):
    """
        Return a new training data after dynamic data augmentation.
        Design for mini-patch CNN model.

        patch - a numpy array, the original training object.
        minb - a number, the minimum of whole seismic section.
        maxb - a number, the maximum of whole seismic section.
        d_transform - a torchvision.tranform function, the type of transform.
        prob - a number, between 0 to 1, the probability of the original training object to be transformed.
    """
    patch = patch.clone()
    p = np.random.uniform()
    if p < prob:
        # for '1-feature' dataset.
        if len(patch.shape) == 2:
            patch = ((255.*(patch-minb))/(maxb-minb)).int()
            img = (d_transform(patch)[0]).float()  # apply the transformation.
            s1 = patch.shape[0]
            s2 = patch.shape[1]
            temp = torch.rand(s1, s2) + torch.rand(s1, s2)*(-1)

            # avoid unexpected bugs from PyTorch
            if torch.max(img) > 100:
                patch = img + temp
            else:
                patch = (255 * img) + temp
            patch = ((patch * (maxb-minb)) / 255.) + minb

        # for '6-feature' dataset.
        if len(patch.shape) == 3:
            for i in range(patch.shape[0]):  
                patch[i] = ((255.*(patch[i]-minb[i]))/(maxb[i]-minb[i])).int()
                img = (d_transform(patch[i])[0]).float()  # apply the transformation.
                s1 = patch[i].shape[0]
                s2 = patch[i].shape[1]
                temp = torch.rand(s1, s2) + torch.rand(s1, s2)*(-1)

                # avoid unexpected bugs from PyTorch
                if torch.max(img) > 100:
                    patch[i] = img + temp
                else:
                    patch[i] = (255 * img) + temp
                patch[i] = ((patch[i] * (maxb[i]-minb[i])) / 255.) + minb[i]
        
    return patch


def evaluate_and_layout_ff(model, data_loader, minib_size):
    """
        Return a predicted class label map on test section.
        Design for feed-forward DNN model.

        model - a PyTorch model.
        data_loader - a PyTorch data loader on test section.
        minib_size - an integer, size of the mini patch of each training object.
    """
    model.eval()  # change the model to evaluation mode.
    y_preds = []
    for X, y in data_loader:  # iterate over test data
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, minib_size)).to(device)   # predict
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]   # apply softmax activation
            y_preds.append(y_pred.cpu().numpy())  # keep track of predicted class
    return np.array(y_preds)


def evaluate_and_layout(model, data_loader, minib_size, channels):
    """
        Return a predicted class label map on test section.
        Design for mini-patch CNN model.

        model - a PyTorch model.
        data_loader - a PyTorch data loader on test section.
        minib_size - an integer, size of the mini patch of each training object.
        channels - an integer, the number of input channels. (1 for '1-feature' and 6 for'6-feature')
    """
    model.eval()  # change the model to evaluation mode.
    y_preds = []
    for X, y in data_loader:  # iterate over test data
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, channels, minib_size, minib_size)).to(device)  # predict
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]  # apply softmax activation
            y_preds.append(y_pred.cpu().numpy())  # keep track of predicted class
    return np.array(y_preds)


def evaluate_and_layout_ed(pred):
    """
        Return a predicted class label map on test section.
        Design for encoder-decoder CNN model.

        pred - a numpy array, the predicted result on test section.
    """
    img = np.zeros_like(pred[0][0])
    for i in range(1, 7):
        for j in range(pred.shape[2]):
            for k in range(pred.shape[3]):
                if pred[0][0][j][k] <= pred[0][i][j][k]:
                    img[j][k] = i  # predict the class label
    return img


def original_size_image(fig_tc, img, xs, xe, ys, ye):
    """
        Return a predicted class label map on test section with the same dimension as the original seismic section.
        Pad the predicted class map with -1.
        Design for encoder-decoder CNN model.

        fig_tc - a numpy array, the class label map of the original test seismic section.
        img - a numpy array, the predicted class label map.
        xs - an integer, the coordinate of the crop starts in the first direction.
        xe - an integer, the coordinate of the crop ends in the first direction.
        ys - an integer, the coordinate of the crop starts in the second direction.
        ye - an integer, the coordinate of the crop ends in the second direction.
    """
    concat_l = []
    concat_r = []
    concat_t = []
    concat_b = []

    # create the padding chunks.
    for i in range(img.shape[0]):
        concat_temp = []
        for j in range(ys):
            concat_temp.append(-1)  # padding with -1 to maintain the result..
        concat_l.append(concat_temp)
    concat_l = np.array(concat_l)
    
    for i in range(img.shape[0]):
        concat_temp = []
        for j in range(ye, fig_tc.shape[1]):
            concat_temp.append(-1)  # padding with -1 to maintain the result.
        concat_r.append(concat_temp)
    concat_r = np.array(concat_r)

    # concatenate the chunks to the original section.
    img = np.concatenate((concat_l, img), axis=1)
    img = np.concatenate((img, concat_r), axis=1)

    # create the padding chunks.
    for i in range(xs):
        concat_temp = []
        for j in range(img.shape[1]):
            concat_temp.append(-1)
        concat_t.append(concat_temp)
    concat_t = np.array(concat_t)
    
    for i in range(xe, fig_tc.shape[0]):
        concat_temp = []
        for j in range(img.shape[1]):
            concat_temp.append(-1)
        concat_b.append(concat_temp)
    concat_b = np.array(concat_b)

    # concatenate the chunks to the original section.
    img = np.concatenate((concat_t, img), axis=0)
    img = np.concatenate((img, concat_b), axis=0)
    
    return img


def accuracy(img, fig_tc):
    """
        Return the accuracy of the predicted class label map on test section corresponding to the real class label map.

        img - a numpy array, the predicted class label map.
        fig_tc - a numpy array, the class label map of the original test seismic section.
    """
    num = img.shape[0] * img.shape[1]
    c_num = 0

    # for '1-feature' dataset.
    if len(fig_tc.shape) == 2:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == fig_tc[i][j]:
                    c_num += 1   # calculate the number of the correct classified pixels.

    # for '6-feature' dataset.
    elif len(fig_tc.shape) == 3:
        fig_new = []
        for j in range(fig_tc.shape[1]):
            fig_newtemp = []
            for k in range(fig_tc.shape[2]):
                for i in range(fig_tc.shape[0]):
                    if fig_tc[i][j][k] == 1:
                        fig_newtemp.append(i)
                        break
            fig_new.append(fig_newtemp)
        fig_tc = np.array(fig_new)
    
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == fig_tc[i][j]:
                    c_num += 1   # calculate the number of the correct classified pixels.
            
    return c_num / num

