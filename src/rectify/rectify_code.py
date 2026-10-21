"""
RECTIFY study processing pipeline.
Adjusted from lung_dce.py

What does it do?

This module processes a colorectal DISCO/DCE-MRI dataset through the following steps:
1. Harmonizes DICOM data structure by merging VFA and DCE series.
2. Computes descriptive DCE maps (e.g., baseline, max enhancement).
3. Automatically segments colorectal organs using TotalSegmentator (or maybe not, totalsegmentator doesn't have a rectal index).
4. Allows manual editing of colon segmentations via napari.  
5. Draws arterial input function (AIF) region on maximum enhancement image.
6. Performs motion correction on VFA and DCE series using mdreg.
7. Fits VFA data to compute R1 and S0 maps.
8. Aligns VFA-derived maps with DCE data space.
9. Conducts ROI-based analysis using both model-free and model-based approaches.
10. Performs pixel-wise pharmacokinetic modeling using model-free and model-based methods.

How to use?

1. Define a virtual envronment and install the requirements.
2. Set the DATAPATH variable to point to the folder containing the DICOM data.
3. Run the script. It will create a build folder in the current working directory.
4. All intermediate results and final outputs will be saved in the build folder

All steps will be executed sequentially when running the script
Steps 1-3 are automated, while steps 4-5 require user interaction. 
The script will pause and open a napari viewer for manual editing 
of segmentations and drawing of AIF. After editing, close the napari 
window to resume execution. All subsequent steps are automated again.
"""

import os

import numpy as np
import dcmri as dc #for dce images
import napari #useful software for viewing and manually editing images (2D, 3D, and 4D)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dbdicom as db #working with DICOM datasets
import miblab #for downloading and uploading data/functions from miblab.org on github
import mdreg #motion correction - particularly useful for non-breathold lung scans and surrounding organs, also cardiac imaging.
import vreg #Facilitate alignment and slicing of 3D (and hence 4D with other packages) datasets with different positions and orientations.
import pydmr #working with .dmr (dynamic MR) data, a dynamic type of csv flie including ROI curve data, pulse sequence parameters, subject properties, and their standard deviations,  
from totalsegmentator.map_to_binary import class_map
from IPython.display import Image, display

from case_studies.utils.stats import mask_stats #masks isolate areas of interest for analysis, e.g. sorting for thresholding. mask_stats gives mask values


def draw_mask(array: np.ndarray, mask=None, name='ROI', contrast_limits=None) -> np.ndarray: #drawing the mask is kept the same: useful for all types of images
    # Ensure it's a numpy array
    array = np.asarray(array).T

    # Initialize mask
    if mask is None:
        mask = np.zeros_like(array, dtype=np.uint8)
    else:
        mask = np.asarray(mask).T

    # Start the napari viewer
    viewer = napari.Viewer()
    viewer.add_image(array, name='Data', blending="additive", contrast_limits=contrast_limits)

    # Add an empty labels layer for drawing
    labels_layer = viewer.add_labels(mask, name=name)

    # Set brush tool as active and radius = 6
    viewer.layers.selection.active = labels_layer
    labels_layer.mode = 'paint'
    labels_layer.brush_size = 6

    # Start the interactive GUI
    napari.run()

    # After closing napari, extract the mask
    mask = (labels_layer.data > 0).astype(np.uint16)

    return mask.T


def draw_mask_notebook(array: np.ndarray, mask=None, name='ROI', contrast_limits=None) -> np.ndarray:
    # This assumes a cell has been run with %gui qt
    # 
    # # Ensure it's a numpy array
    array = np.asarray(array).T

    # Initialize mask
    if mask is None:
        mask = np.zeros_like(array, dtype=np.uint8)
    else:
        mask = np.asarray(mask).T

    # Start the napari viewer
    viewer = napari.Viewer()
    viewer.add_image(array, name='Data', blending="additive", contrast_limits=contrast_limits)

    # Add an empty labels layer for drawing
    labels_layer = viewer.add_labels(mask, name=name)

    # Set brush tool as active and radius = 6
    viewer.layers.selection.active = labels_layer
    labels_layer.mode = 'paint'
    labels_layer.brush_size = 6

    # Block execution strictly until THIS viewer window is closed
    viewer.show(block=True)

    # After closing napari, extract the mask
    mask = (labels_layer.data > 0).astype(np.uint16)

    # Explicitly close and clean up viewer memory/Qt resources
    viewer.close()

    return mask.T


def harmonize_data(datapath, data):

    # If the outputs exists, delete them first
    vfa_series = data + ['3D-VFA']
    dce_series = data + ['3D-DCE'] #vfa and dce data both used in rectify
    db.delete(vfa_series, not_exists_ok=True)
    db.delete(dce_series, not_exists_ok=True)

    # Merge VFA into 1 series to simplify analysis
    # Write the correct flip angle in each series
    for fa in [2,4,10,20,30]: #flip angles used in RECTIFY 
        fa_series = db.series(datapath, contains=f'flip {fa}')[0]
        fa_series_copy = db.copy(fa_series)
        db.edit(fa_series_copy, {'FlipAngle':fa})
        db.move(fa_series_copy, vfa_series) # bug not removing empty folders

    # Merge DCE into one single series
    dce = db.series(datapath, contains='WITH CONTRAST')
    [db.copy(phase, dce_series) for phase in dce]
    # Store timing info in acquisition time in seconds (miblab default)
    acq_time = db.values(dce_series, 'TriggerTime', dims=['SliceLocation', 'TriggerTime'])
    db.edit(dce_series, {'AcquisitionTime': acq_time/1000}, dims=['SliceLocation', 'TriggerTime'])

    print('Finished harmonizing data. VFA and DCE series are now merged into single series for analysis.')


def view_dce_notebook(data):
    # This assumes a cell has been run with %gui qt

    # img, mask = 'S0_vfa_on_dce', "lung_left"
    # vol = db.volume(MAPS + [img])
    # mask_vol = db.volume(MASKS + [mask])
    # draw_mask(vol.values, mask_vol.values, name=mask, contrast_limits=[0, 5000])

    map = '3D-DCE'
    vol = db.volume(data + [map], 'AcquisitionTime')
    viewer = napari.Viewer()
    viewer.add_image(vol.values.T, name=map, contrast_limits=[0, 1000])

def compute_descriptive_maps(data, maps):

    # Compute maps of descriptive parameters and save
    dce_series = data + ['3D-DCE']
    dce_data = db.volume(dce_series, 'AcquisitionTime')
    dce_maps = dc.describe(dce_data.values)
    for par in dce_maps:
        par_vol = (dce_maps[par], dce_data.affine)
        par_series = maps + [par]
        db.delete(par_series, not_exists_ok=True) 
        db.write_volume(par_vol, par_series, ref=dce_series)

    print('Finished computing descriptive maps. Maps are saved in the MAPS series.')


def autosegment_colorectal(maps, masks): #comment out the colorectal section: go to section 3 for colorectal orgns

    # --- 1. Draw colorectal on baseline image
    #ref_series = maps + ['Sb']
    #ref_vol = db.volume(ref_series) 

    # Create label and save
    #label_vol = miblab.totseg(ref_vol, cutoff=0.01, task='total_mr', device='cpu')
    #total_mr_series = masks + ['total_mr']
    #db.delete(total_mr_series, not_exists_ok=True) 
    #db.write_volume(label_vol, total_mr_series, ref=ref_series)

    # Create individual masks for key structures
    #total_mr = class_map['total_mr']
    #for label_value in [10,11]:
    #    mask_arr = np.where(label_vol.values==label_value, 1, 0).astype(np.int16)
    #    mask_series = masks + [total_mr[label_value]]
    #    db.delete(mask_series, not_exists_ok=True) 
    #    db.write_volume((mask_arr, label_vol.affine), mask_series, ref=ref_series)

    #print('Finished automatic segmentation of colorectal. Masks are saved in the MASKS series.')

    # # --- 2. Draw liver/kidneys/vessels on maximum enhancement image
    # ref_series = MAPS + ['SEmax']
    # ref_vol = db.volume(ref_series) 

    # # Create label and save
    # label_vol = miblab.totseg(ref_vol, cutoff=0.01, task='total_mr', device='cpu')
    # db.write_volume(label_vol, MASKS + ['total_mr_SEmax'], ref=ref_series)

    # # Create individual masks for key structures
    # for label_value in [2,3,5,22,23,24,25]:
    #     mask_arr = np.where(label_vol.values==label_value, 1, 0).astype(np.int16)
    #     db.write_volume((mask_arr, label_vol.affine), MASKS + [TOTAL_MR[label_value]], ref=ref_series)

# --- 3. Draw colorectal organs on baseline image
    ref_series = maps + ['Sb']
    ref_vol = db.volume(ref_series) 

    # Create label and save
    label_vol = miblab.totseg(ref_vol, cutoff=0.01, task='total_mr', device='cpu')
    total_mr_series = masks + ['total_mr']
    db.delete(total_mr_series, not_exists_ok=True) 
    db.write_volume(label_vol, total_mr_series, ref=ref_series)

    # Create individual masks for key structures
    total_mr = class_map['total_mr']
    for label_value in [20]: #index for colon RECTUM NOT INCLUDED
        mask_arr = np.where(label_vol.values==label_value, 1, 0).astype(np.int16)
        mask_series = masks + [total_mr[label_value]]
        db.delete(mask_series, not_exists_ok=True) 
        db.write_volume((mask_arr, label_vol.affine), mask_series, ref=ref_series)

    print('Finished automatic segmentation of colorectal system. Masks are saved in the MASKS series.')


def edit_autosegmented_colorectal(maps, masks):

    # Edit colorectal on baseline image
    ref_series = maps + ['Sb']
    ref_vol = db.volume(ref_series) 

    # Edit  ROIs
    masks_to_edit = ['colon']
    for mask in masks_to_edit:
        mask_array = db.volume(masks + [mask]).values
        mask_array = draw_mask(ref_vol.values, mask=mask_array, name=mask)
        mask_vol = (mask_array, ref_vol.affine)
        mask_series = masks + [f"{mask}_edited"]
        db.delete(mask_series, not_exists_ok=True)
        db.write_volume(mask_vol, mask_series, ref=ref_series)

    print('Finished editing colon segmentations. Edited masks are saved in the MASKS series.')


def edit_autosegmented_colorectal_notebook(maps, masks):
    # Note this may also work in the script, just has not been tested
    # If it works, can remove the other function and just use this one for both script and notebook

    # Edit colorectal on baseline image
    ref_series = maps + ['Sb']
    ref_vol = db.volume(ref_series) 

    # Edit colon ROIs
    masks_to_edit = ['colon']
    for mask in masks_to_edit:
        mask_array = db.volume(masks + [mask]).values
        mask_array = draw_mask_notebook(ref_vol.values, mask=mask_array, name=mask)
        mask_vol = (mask_array, ref_vol.affine)
        mask_series = masks + [f"{mask}_edited"]
        db.delete(mask_series, not_exists_ok=True)
        db.write_volume(mask_vol, mask_series, ref=ref_series)

    print('Finished editing colon segmentations. Edited masks are saved in the MASKS series.')


def draw_arterial_inputs(maps, masks):

    # Draw right heart and pulmonary artery on SEmax
    ref_series = maps + ['SEmax']
    ref_vol = db.volume(ref_series) 

    # # Draw right heart for mdreg
    # # Not used at this stage
    # roi = 'input'
    # mask_array = draw_mask(ref_vol.values, name=roi)  
    # mask_vol = (mask_array, ref_vol.affine)
    # mask_series = MASKS + [roi]
    # db.delete(mask_series, not_exists_ok=True)
    # db.write_volume(mask_vol, mask_series, ref=ref_series)

    # Draw AIF
    roi = 'pulmonary_artery'
    mask_array = draw_mask(ref_vol.values, name=roi)  
    mask_vol = (mask_array, ref_vol.affine)
    mask_series = masks + [roi]
    db.delete(mask_series, not_exists_ok=True)
    db.write_volume(mask_vol, masks + [roi], ref=ref_series)

def draw_arterial_inputs_notebook(maps, masks):

    # Draw right heart and pulmonary artery on SEmax
    ref_series = maps + ['SEmax']
    ref_vol = db.volume(ref_series) 

    # # Draw right heart for mdreg
    # # Not used at this stage
    # roi = 'input'
    # mask_array = draw_mask(ref_vol.values, name=roi)  
    # mask_vol = (mask_array, ref_vol.affine)
    # mask_series = MASKS + [roi]
    # db.delete(mask_series, not_exists_ok=True)
    # db.write_volume(mask_vol, mask_series, ref=ref_series)

    # Draw AIF
    roi = 'pulmonary_artery'
    mask_array = draw_mask_notebook(ref_vol.values, name=roi)  
    mask_vol = (mask_array, ref_vol.affine)
    mask_series = masks + [roi]
    db.delete(mask_series, not_exists_ok=True)
    db.write_volume(mask_vol, masks + [roi], ref=ref_series)


def vfa_motion_correction(data, maps):
    moco_series = maps + ['3D-VFA-MOCO']
    fit_series = maps + ['3D-VFA-MOCO-FIT']
    db.delete(moco_series, not_exists_ok=True)
    db.delete(fit_series, not_exists_ok=True)

    # Get data
    vfa_series = data + ['3D-VFA']
    vfa_vol = db.volume(vfa_series, 'FlipAngle')
    
    # Perform motion correction
    coreg, fit, transfo, pars = mdreg.fit(
        vfa_vol.values, 
        fit_image = {
            'func': mdreg.fit_spgr_vfa_lin,
            'FA': vfa_vol.coords[0], 
            'parallel': False,
            'progress_bar': True,                          
        }, 
        fit_coreg = {
            'package': 'ants',
            'type_of_transform': 'SyNOnly',
        }, 
        maxit = 5,       
        verbose = 2,
    )

    # # Compute R1
    # tr_sec = db.unique('RepetitionTime', vfa_series)[0]/1000 
    # R1 = - np.log(pars[...,1]) / tr_sec

    # Save results to DICOM
    vol_coreg = vreg.volume(coreg, vfa_vol.affine, coords=vfa_vol.coords, dims=vfa_vol.dims)
    vol_fit = vreg.volume(fit, vfa_vol.affine, coords=vfa_vol.coords, dims=vfa_vol.dims)
    db.write_volume(vol_coreg, moco_series, ref=vfa_series)
    db.write_volume(vol_fit, fit_series, ref=vfa_series)


def view_vfa_notebook(maps):
    # This assumes a cell has been run with %gui qt
    map = '3D-VFA-MOCO'
    vol = db.volume(maps + [map], 'FlipAngle')
    viewer = napari.Viewer()
    viewer.add_image(vol.values.T, name=map, contrast_limits=[0, 1000])


def vfa_fit(maps):
    R1_series = maps + ['R1_vfa']
    S0_series = maps + ['S0_vfa']
    db.delete(R1_series, not_exists_ok=True)
    db.delete(S0_series, not_exists_ok=True)
    
    # Get input data
    vfa_series = maps + ['3D-VFA-MOCO']
    vfa_volume = db.volume(vfa_series, 'FlipAngle')
    tr_sec = db.unique('RepetitionTime', vfa_series)[0]/1000 
    vfa_array = vfa_volume.values
    fa_deg = vfa_volume.coords[0]
    S0_max = 10 * np.amax(vfa_array) / np.sin(np.deg2rad(np.min(fa_deg)))
    R1_max = 10 # sec
    bounds = ([0,0], [R1_max, S0_max])
    
    # Compute linear and save results
    R1_array, S0_array = dc.vfa_linear(vfa_array, fa_deg, tr_sec, bounds, verbose=0)
    db.write_volume((R1_array, vfa_volume.affine), R1_series, ref=vfa_series)
    db.write_volume((S0_array, vfa_volume.affine), S0_series, ref=vfa_series)

    # # Compute nonlinear and save results
    # R1_array, S0_array = dc.vfa_nonlinear(vfa_array, fa_deg, tr_sec, bounds, verbose=0)
    # db.write_volume((R1_array, vfa_volume.affine), MAPS + ['R1_vfa_nonlin'], ref=vfa_series)
    # db.write_volume((S0_array, vfa_volume.affine), MAPS + ['S0_vfa_nonlin'], ref=vfa_series)


# def dce_fit():

#     # Get data
#     dce_series = DATA + ['3D-DCE']
#     dce_vol = db.volume(dce_series, 'TriggerTime')

#     n_comp = 8
#     fit, pars = mdreg.fit_pca(dce_vol.values, n_comp)  
#     error = np.linalg.norm(fit-dce_vol.values, axis=-1)

#     # Save results to DICOM
#     vol_comp = vreg.volume(pars, dce_vol.affine, coords=[np.arange(n_comp)], dims=['TriggerTime'])
#     vol_fit = vreg.volume(fit, dce_vol.affine, coords=dce_vol.coords, dims=dce_vol.dims)
#     db.write_volume(vol_comp, MAPS + [f'3D-DCE-PCA-COMP_{n_comp}'], ref=dce_series)
#     db.write_volume(vol_fit, MAPS + [f'3D-DCE-PCA-FIT_{n_comp}'], ref=dce_series)
#     db.write_volume((error, dce_vol.affine), MAPS + [f'3D-DCE-PCA-ERROR_{n_comp}'], ref=dce_series)


def dce_motion_correction(data, maps):

    moco_series = maps + ['3D-DCE-MOCO']
    db.delete(moco_series, not_exists_ok=True)

    # Get data
    dce_series = data + ['3D-DCE']
    dce_vol = db.volume(dce_series, 'AcquisitionTime')

    # OPTION 1: DO NOTHING
    db.write_volume(dce_vol, moco_series, ref=dce_series)

    # OPTION 2: MDREG - requires more optimization and evaluation on moving data.

    # aif_mask = db.volume(MASKS + ['input']).values
    # time = dce_vol.coords[0]/1000
    # time = time - time[0]
    # aif_signal = np.mean(dce_vol.values[aif_mask > 0, :], axis=0)

    # # Model fitting options
    # fit_pca = {
    #     'func': mdreg.fit_pca,
    #     'n_components': 8,
    # }
    # fit_2cm = {
    #     'func': mdreg.fit_2cm_lin,
    #     'time': time,
    #     'aif': aif_signal,
    #     'baseline': 1,
    #     'input_corr': True,
    # }
    # fit_deconv = {
    #     'func': mdreg.fit_deconvolution,
    #     'aif': aif_signal,
    #     'n0': 1,
    #     'tol': 0.2,
    # }

    # # Coregistration options
    # fit_ants = {
    #     'package': 'ants',
    #     'type_of_transform': 'SyNOnly',
    #     'parallel': False,
    #     'progress_bar': True,  
    # }
    # fit_skimage = {
    #     'package': 'skimage',
    #     'attachment': 10,
    #     'parallel': False,
    #     'progress_bar': True,  
    # }
    
    # # Perform motion correction
    # coreg, fit, _, _ = mdreg.fit(
    #     dce_vol.values, fit_image=fit_pca, fit_coreg=fit_skimage, 
    #     maxit = 5, verbose = 2,
    # )

    # # Save results to DICOM
    # vol_coreg = vreg.volume(coreg, dce_vol.affine, coords=dce_vol.coords, dims=dce_vol.dims)
    # vol_fit = vreg.volume(fit, dce_vol.affine, coords=dce_vol.coords, dims=dce_vol.dims)
    # db.write_volume(vol_coreg, MAPS + ['3D-DCE-MOCO'], ref=dce_series)
    # db.write_volume(vol_fit, MAPS + ['3D-DCE-MOCO-FIT'], ref=dce_series)
      

def align_vfa_with_dce(maps, masks):

    S0_series = maps + ['S0_vfa_on_dce']
    R1_series = maps + ['R1_vfa_on_dce']
    db.delete(S0_series, not_exists_ok=True)
    db.delete(R1_series, not_exists_ok=True)

    # Slice R1 and S0 on DCE
    dce_Sb_vol = db.volume(maps + ['Sb'])
    vfa_S0_vol = db.volume(maps + ['S0_vfa'])
    vfa_R1_vol = db.volume(maps + ['R1_vfa'])
    vfa_S0_on_dce_vol = vfa_S0_vol.slice_like(dce_Sb_vol)
    vfa_R1_on_dce_vol = vfa_R1_vol.slice_like(dce_Sb_vol)

    # Compute DCE S0 with R1 map
    dce_series = maps + ['3D-DCE-MOCO']
    pars = db.unique(['FlipAngle','RepetitionTime'], dce_series)
    Sref = dc.signal_ss(1, vfa_R1_on_dce_vol.values, pars['RepetitionTime'][0]/1000, pars['FlipAngle'][0])
    dce_S0 = np.divide(dce_Sb_vol.values, Sref, out=np.zeros_like(Sref, dtype=float), where=Sref!=0)

    # Rescale vfa S0 to dce S0 to correct for recalibration between vfa and dce
    liver_mask = db.volume(masks + ['total_mr']).values == 5
    liver_vfa_S0 = np.mean(vfa_S0_on_dce_vol.values[liver_mask])
    liver_dce_S0 = np.mean(dce_S0[liver_mask])
    vfa_S0_on_dce_scaled = vfa_S0_on_dce_vol.values * liver_dce_S0 / liver_vfa_S0

    # Clip S0 maps at 10 x the liver values
    S0_max = 100 * liver_dce_S0
    dce_S0[dce_S0 >  S0_max] = S0_max
    vfa_S0_on_dce_scaled[vfa_S0_on_dce_scaled > S0_max] = S0_max

    # Save S0 and R1 maps
    # db.write_volume((dce_S0, dce_Sb_vol.affine), MAPS + ['S0_dce'], ref=dce_series)
    db.write_volume((vfa_S0_on_dce_scaled, dce_Sb_vol.affine), S0_series, ref=dce_series)
    db.write_volume((vfa_R1_on_dce_vol.values, dce_Sb_vol.affine), R1_series, ref=dce_series)


def roi_analysis_model_free(maps, masks, imagepath):

    dce_series = maps + ['3D-DCE-MOCO']
    dce_data = db.volume(dce_series, 'AcquisitionTime')
    aif_mask = db.volume(masks + ['pulmonary_artery']).values
    R10_map = db.volume(maps + ['R1_vfa_on_dce']).values
    S0_map = db.volume(maps + ['S0_vfa_on_dce']).values

    time = dce_data.coords[0]
    time = time - time[0]
    aif_signal = np.mean(dce_data.values[aif_mask > 0, :], axis=0)
    pars = db.unique(['FlipAngle','RepetitionTime'], dce_series)

    png_file = {}
    for roi in ['colon_edited']:

        colon_mask = db.volume(masks + [roi]).values
        colon_signal = np.mean(dce_data.values[colon_mask > 0, :], axis=0)

        colon = dc.TissueLS(
            sequence = 'SS', 
            dt = time[1], 
            TR = pars['RepetitionTime'][0]/1000,
            FA = pars['FlipAngle'][0],
            r1 = dc.relaxivity(3, 'blood', 'gadobutrol'), 
            R10 = np.mean(R10_map[colon_mask > 0]),
            S0 = np.mean(S0_map[colon_mask > 0]),
            R10a = np.mean(R10_map[aif_mask > 0]),
            S0a = np.mean(S0_map[aif_mask > 0]),
        )
        colon.train(colon_signal, aif_signal, tol=0.15, init_s0=False)

        png_file[roi] = os.path.join(imagepath, f'model_free_{roi}.png')
        colon.plot(colon_signal, round_to=3, show=False, fname=png_file[roi])

    return png_file


def roi_analysis_model_based(maps, masks, imagepath):

    dce_series = maps + ['3D-DCE-MOCO']
    dce_data = db.volume(dce_series, 'AcquisitionTime')
    aif_mask = db.volume(masks + ['pulmonary_artery']).values
    R10_map = db.volume(maps + ['R1_vfa_on_dce']).values
    S0_map = db.volume(maps + ['S0_vfa_on_dce']).values

    time = dce_data.coords[0]
    time = time - time[0]
    aif_signal = np.mean(dce_data.values[aif_mask > 0, :], axis=0)
    pars = db.unique(['FlipAngle','RepetitionTime'], dce_series)

    png_file = {}
    for roi in ['colon_edited']:

        colon_mask = db.volume(masks + [roi]).values
        colon_signal = np.mean(dce_data.values[colon_mask > 0, :], axis=0)

        # Perform the analysis
        colon = dc.Tissue(
            aif = aif_signal, 
            t = time, 
            kinetics = 'NXP', 
            TR = pars['RepetitionTime'][0]/1000,
            FA = pars['FlipAngle'][0],
            r1 = dc.relaxivity(3, 'blood', 'gadobutrol'), 
            R10 = np.mean(R10_map[colon_mask > 0]),
            S0 = np.mean(S0_map[colon_mask > 0]),
            R10a = np.mean(R10_map[aif_mask > 0]),
            S0a = np.mean(S0_map[aif_mask > 0]),
            n0 = 5,
            free = {'Fb':[0,1], 'vb':[0,1]}, 
        )
        colon.train(time, colon_signal, init_s0=False)

        png_file[roi] = os.path.join(imagepath, f'model_based_{roi}.png')
        colon.plot(time, colon_signal, round_to=3, show=False, fname=png_file[roi])

    return png_file


def pixel_analysis_model_free(maps, masks, imagepath):

    # If the outputs exists, delete them first
    Fb_series = maps + ['Fb_free']
    ve_series = maps + ['ve_free']
    db.delete(Fb_series, not_exists_ok=True)
    db.delete(ve_series, not_exists_ok=True)

    # Read data volumes
    dce_series = maps + ['3D-DCE-MOCO']
    dce_data = db.volume(dce_series, 'AcquisitionTime')
    aif_mask = db.volume(masks + ['pulmonary_artery']).values > 0
    R10_map = db.volume(maps + ['R1_vfa_on_dce']).values
    S0_map = db.volume(maps + ['S0_vfa_on_dce']).values

    # Read signals and sequence parameters
    time = dce_data.coords[0]
    aif_signal = np.mean(dce_data.values[aif_mask, :], axis=0)
    pars = db.unique(['FlipAngle','RepetitionTime'], dce_series)

    # Train a tissue model
    body = dc.TissueLSArray(
        dce_data.shape[:3],
        sequence = 'SS', 
        dt = time[1] - time[0], 
        TR = pars['RepetitionTime'][0]/1000,
        FA = pars['FlipAngle'][0],
        r1 = dc.relaxivity(3, 'blood', 'gadobutrol'), 
        R10 = R10_map,  
        S0 = S0_map,
        R10a = np.mean(R10_map[aif_mask]),
        S0a = np.mean(S0_map[aif_mask]),
    )
    body.train(dce_data.values, aif_signal, tol=0.15, init_s0=False)
    
    # vol_conc = vreg.volume(body.predict_conc(), dce_data.affine, coords=dce_data.coords, dims=dce_data.dims)
    # db.write_volume(vol_conc, MAPS + ['C'+ext], ref=dce_series)
    pars = body.params()
    db.write_volume((pars['Fb'], dce_data.affine), Fb_series, ref=dce_series)
    db.write_volume((pars['ve'], dce_data.affine), ve_series, ref=dce_series)
    #db.write_volume((pars['S0'], dce_data.affine), MAPS + ['S0_free'+ext], ref=dce_series)

    colon_mask = db.volume(masks + ['colon']) 
    #lung_left_mask = db.volume(masks + ['lung_left'])
    #lung_mask = (lung_right_mask.values==1) | (lung_left_mask.values==1)
    vmin = {'Fb':0, 've':0, 'background':0}
    vmax = {'Fb':0.5, 've':1.0, 'background':np.percentile(S0_map, 99)}

    png_file = os.path.join(imagepath, f'model_free.png')
    body.plot_overlay(colon_mask, vmin=vmin, vmax=vmax, show=False, fname=png_file)

    return (
        os.path.join(imagepath, f'Fb_model_free.png'), 
        os.path.join(imagepath, f've_model_free.png'),
    )


def pixel_analysis_model_based(maps, masks, imagepath, checkpointspath):

    Fb_series = maps + ['Fb_model']
    vb_series = maps + ['vb_model']
    db.delete(Fb_series, not_exists_ok=True)
    db.delete(vb_series, not_exists_ok=True)

    # Read data volumes
    dce_series = maps + ['3D-DCE-MOCO']
    dce_data = db.volume(dce_series, 'AcquisitionTime')
    aif_mask = db.volume(masks + ['pulmonary_artery']).values > 0
    R10_map = db.volume(maps + ['R1_vfa_on_dce']).values
    S0_map = db.volume(maps + ['S0_vfa_on_dce']).values
    #lung_right_mask = db.volume(masks + ['lung_right']).values
    #lung_left_mask = db.volume(masks + ['lung_left']).
    colon_mask = db.volume(masks + ['colon_edited']).values > 0

    # Get signals, sequence parameters and masks
    time = dce_data.coords[0]
    time = time - time[0]
    aif_signal = np.mean(dce_data.values[aif_mask, :], axis=0)
    pars = db.unique(['FlipAngle','RepetitionTime'], dce_series)
    #lung_mask = (lung_right_mask==1) | (lung_left_mask==1)
    dce_colon = dce_data.values[colon_mask, :]

    # # Read
    # vmin = {'Fb':0, 've':0, 'background':0}
    # vmax = {'Fb':0.5, 've':1.0, 'background':np.percentile(S0_map, 99)}
    # Fb = db.volume(MAPS + ['Fb_model']).values
    # ve = db.volume(MAPS + ['ve_model']).values
    # lung = dc.TissueArray(dce_data.shape[:3], kinetics='FX', sequence='SS', Fb=Fb, ve=ve, R10=R10_map, S0=S0_map).load(os.path.join(checkpointspath, f'model_3d'))
    # lung.plot_overlay(dce_data.values, lung_mask, vmin=vmin, vmax=vmax, show=False, fname=os.path.join(imagepath, f'model_based.png'))

    # return

    model_pars = {
        'aif': aif_signal, 
        't': time, 
        'TR': pars['RepetitionTime'][0]/1000,
        'FA': pars['FlipAngle'][0],
        'r1': dc.relaxivity(3, 'blood', 'gadobutrol'), 
        'R10': R10_map[colon_mask], 
        'S0': S0_map[colon_mask],
        'R10a': np.mean(R10_map[aif_mask]),
        'S0a': np.mean(S0_map[aif_mask]),
        'n0': 3,
        'parallel': False,
        'verbose': 1,   
        'free': {'Fb':[0,1], 'vb':[0,1]},
    }
   
    # Train a tissue model (scale by max)
    colon_model = dc.TissueArray(dce_colon.shape[:1], kinetics='NXP', sequence='SS', **model_pars)
    colon_model.train(time, dce_colon, init_s0=False)
    colon_model.save(os.path.join(checkpointspath, f'model_flat'))

    # Create maps
    vb = np.zeros(dce_data.shape[:3])
    Fb = np.zeros(dce_data.shape[:3])

    vb[colon_mask] = colon_model.params('vb')
    Fb[colon_mask] = colon_model.params('Fb')

    # Save results
    db.write_volume((vb, dce_data.affine), vb_series, ref=dce_series)
    db.write_volume((Fb, dce_data.affine), Fb_series, ref=dce_series)

    # Load again with 3D maps
    vmin = {'Fb':0, 'vb':0, 'background':0}
    vmax = {'Fb':0.5, 'vb':1.0, 'background':np.percentile(S0_map, 99)}
    model_pars['R10'] = R10_map
    model_pars['S0'] = S0_map
    colon_model = dc.TissueArray(dce_data.shape[:3], kinetics='NXP', sequence='SS', Fb=Fb, vb=vb, **model_pars)
    png_file = os.path.join(imagepath, f'model_based.png')
    colon_model.plot_overlay(dce_data.values, colon_mask, vmin=vmin, vmax=vmax, show=False, fname=png_file)
    colon_model.save(os.path.join(checkpointspath, f'model_3d'))

    return (
        os.path.join(imagepath, f'Fb_model_based.png'), 
        os.path.join(imagepath, f've_model_based.png'),
    )



    

def view_maps(data):

    # img, mask = 'S0_vfa_on_dce', "lung_left"
    # vol = db.volume(MAPS + [img])
    # mask_vol = db.volume(MASKS + [mask])
    # draw_mask(vol.values, mask_vol.values, name=mask, contrast_limits=[0, 5000])

    map = '3D-DCE'
    vol = db.volume(data + [map], 'AcquisitionTime')
    viewer = napari.Viewer()
    viewer.add_image(vol.values.T, name=map, contrast_limits=[0, 1000])
    napari.run()


def view_png(png_file):
    display(Image(filename=png_file))

    # img = mpimg.imread(png_file)
    # plt.figure(dpi=300)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()


def pixel_averages_model_based(maps, masks, subj, build_path):

    fb_image = db.volume(maps + ['Fb_model']).values
    vb_image = db.volume(maps + ['vb_model']).values
    #lung_right_mask = db.volume(masks + ['lung_right']).values
    colon_mask = db.volume(masks + ['colon']).values

    fb_colon = mask_stats(fb_image, colon_mask)
    vb_colon = mask_stats(vb_image, colon_mask)

    data = {}
    #data = data | {f"right_lung-fb-{k}":[f"Right lung blood flow {k}", 'mL/sec/cm3', 'float'] for k in fb_colon.keys()}
    #data = data | {f"right_lung-vb-{k}":[f"Right lung blood volume {k}", 'mL/cm3', 'float'] for k in vb_colon.keys()}
    data = data | {f"colon-fb-{k}":[f"colond flow {k}", 'mL/sec/cm3', 'float'] for k in fb_colon.keys()}
    data = data | {f"colon-vb-{k}":[f"Colon blood volume {k}", 'mL/cm3', 'float'] for k in vb_colon.keys()}

    pars = {}
    #pars = pars | {(subj, 'DCE-MRI', f"right_lung-fb-{k}"):v for k, v in fb_colon.items()}
    #pars = pars | {(subj, 'DCE-MRI', f"right_lung-vb-{k}"):v for k, v in vb_colon.items()}
    pars = pars | {(subj, 'DCE-MRI', f"colon-fb-{k}"):v for k, v in fb_colon.items()}
    pars = pars | {(subj, 'DCE-MRI', f"colon-vb-{k}"):v for k, v in vb_colon.items()}

    dmr_file = os.path.join(build_path, 'roi_vals')
    dmr = {'data':data, 'pars':pars}
    pydmr.write(dmr_file, dmr)


if __name__=='__main__':

    # --- USER SETTINGS ---
    #
    DATAPATH = r"C:\Users\cmq26nc\neoadj\venvs\rectify\RECTIFY_01\scans" #change RECTIFY01 to RECTIFY_XY for each patient
    BUILD_PATH = r"C:\Users\cmq26nc\neoadj\venvs\rectify\RECTIFY_01\Rectify" #need to change this to a new folder for each run
    # 
    # --- END USER SETTINGS ---

    build_path = BUILD_PATH
    resultspath = os.path.join(build_path, 'dicom')
    IMAGEPATH = os.path.join(build_path, 'images')
    CHECKPOINTPATH = os.path.join(build_path, 'checkpoints')
    os.makedirs(IMAGEPATH, exist_ok=True)
    os.makedirs(CHECKPOINTPATH, exist_ok=True)

    SUBJ = 'SheffieldSubject'
    DATA = [resultspath, SUBJ, 'Data']
    MAPS = [resultspath, SUBJ, 'Maps']
    MASKS = [resultspath, SUBJ, 'Masks']
    TOTAL_MR = class_map['total_mr']

    harmonize_data(DATAPATH, DATA)
    compute_descriptive_maps(DATA, MAPS)
    autosegment_colorectal(MAPS, MASKS)
    edit_autosegmented_colorectal(MAPS, MASKS)
    draw_arterial_inputs(MAPS, MASKS)
    vfa_motion_correction(DATA, MAPS)
    vfa_fit(MAPS)
    dce_motion_correction(DATA, MAPS)
    align_vfa_with_dce(MAPS, MASKS)
    roi_analysis_model_based(MAPS, MASKS, IMAGEPATH)
    roi_analysis_model_free(MAPS, MASKS, IMAGEPATH)
    pixel_analysis_model_free(MAPS, MASKS, IMAGEPATH)
    pixel_analysis_model_based(MAPS, MASKS, IMAGEPATH, CHECKPOINTPATH)
    pixel_averages_model_based(MAPS, MASKS, SUBJ, BUILD_PATH)

    # view_maps(DATA)
    