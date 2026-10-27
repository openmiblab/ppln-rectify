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
import json
from collections import defaultdict

import numpy as np
import dcmri as dc #for dce images
import napari #useful software for viewing and manually editing images (2D, 3D, and 4D)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dbdicom as db #working with DICOM datasets
import dbdicom.dataset as dbdataset
import miblab #for downloading and uploading data/functions from miblab.org on github
import mdreg #motion correction - particularly useful for non-breathold lung scans and surrounding organs, also cardiac imaging.
import vreg #Facilitate alignment and slicing of 3D (and hence 4D with other packages) datasets with different positions and orientations.
import pydicom
import pydmr #working with .dmr (dynamic MR) data, a dynamic type of csv flie including ROI curve data, pulse sequence parameters, subject properties, and their standard deviations,  
from totalsegmentator.map_to_binary import class_map
from IPython.display import Image, display

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

    fa_series = {}
    missing_flip_angles = []
    for fa in [2, 4, 10, 20, 30]:
        matches = db.series(datapath, contains=f'flip {fa}')
        if matches:
            fa_series[fa] = matches[0]
        else:
            missing_flip_angles.append(fa)

    if missing_flip_angles: # applies if even one perfusion flip angle is missing, useful check with fa 4 having a slightly different file name than the others (missing an underscore before the number)
        raise ValueError(
            f"No VFA series found for flip angle(s): {missing_flip_angles}."
        )

    dce = db.series(datapath, contains='WITH CONTRAST')
    if not dce:
        dce = db.series(datapath, contains='DISCO')
    if not dce:
        raise ValueError(
            "No DCE series found. Expected a series name containing "
            "'WITH CONTRAST' or 'DISCO'." #change this, add a separate one for disco
        )

    with open(os.path.join(datapath, 'dbtree.json'), encoding='utf-8') as f:
        dbtree = json.load(f)

    def source_series_label(series):
        source_files = db.register.files(dbtree, series)
        if not source_files:
            raise ValueError(f"No DICOM files found for series: {series}.")
        series_description = series[-1][0]
        source_folder = os.path.relpath(source_files[0], datapath).split(os.sep, 1)[0]
        return (
            f"SeriesDescription: {series_description}; "
            f"source folder: {source_folder}"
        )

    # If the outputs exists, delete them first
    vfa_series = data + ['3D-VFA']
    dce_series = data + ['3D-DCE'] #vfa and dce data both used in rectify
    db.delete(vfa_series, not_exists_ok=True)
    db.delete(dce_series, not_exists_ok=True)

    # Merge VFA into 1 series to simplify analysis
    # Write the correct flip angle in each series
    for fa, source_series in fa_series.items():
        staged_series = data + [f'3D-VFA-{fa}']
        db.delete(staged_series, not_exists_ok=True)
        print(f"Merging {source_series_label(source_series)}")
        db.copy(source_series, staged_series)
        db.edit(staged_series, {'FlipAngle':fa})
        db.move(staged_series, vfa_series)

    # Merge DCE into one single series
    for phase in dce:
        print(f"Merging {source_series_label(phase)}")
        db.copy(phase, dce_series)
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

#Defining log fit used in ADC equation, ADC=-(1/b_1-b_0) * log(S_1/S_0)
#Rearrange to S_1=S_0*e^(-b*ADC) so the slope of log(S_1/S_0) against the b-values gives -ADC
def _fit_adc(signal, b_values):
    b_values = np.asarray(b_values, dtype=float) #convert b-values to numpy arrays
    if signal.shape[-1] != b_values.size or b_values.size < 2: #check that the number of signal values matches the number of b-values and theres at least 2 b-values
        raise ValueError("ADC fitting requires at least two signal values with matching b-values.")
    if np.any(~np.isfinite(b_values)) or np.any(b_values < 0): #rejects NAN, infinite, or neg b-values
        raise ValueError("Diffusion b-values must be finite and non-negative.")

    valid = np.all(np.isfinite(signal) & (signal > 0), axis=-1) #make boolean mask for whether locations have only finite, positive signals
    log_signal = np.zeros(signal.shape, dtype=float) #create an array to put log(signal) the same shape as the signal array
    np.log(signal, out=log_signal, where=np.isfinite(signal) & (signal > 0)) #ignoring NAN, infinite, and negative values so logarithm is mathematicallly possible, performs the logarithmic transformation
    centered_b = b_values - np.mean(b_values) #b_1-b_0
    denominator = np.sum(centered_b**2) #sets (b_1-b_0)**2 as the denominator for linear regression
    if denominator == 0: #check for different b-values
        raise ValueError("ADC fitting requires at least two distinct b-values.")

    slope = np.sum((log_signal - np.mean(log_signal, axis=-1, keepdims=True))* centered_b, axis=-1,) / denominator #performing the equation from previous variables, 
    adc = np.zeros(signal.shape[:-1], dtype=np.float32) #makes ADC output array, shape[-1] denotes the last dimension of the adc_map array (rows, columns, slices, b-value measurements) so returns rows, colmns,and slices but not b-values
    adc[valid] = np.maximum(-slope[valid], 0) #sets negative ADC values to 0
    return adc

#Apparent diffusion coefficient maps for dwi images against T2 images
def adc_map(datapath, maps):
    """Create an ADC map from the axial DWI series in its native image space."""
    dwi_series = db.series(datapath, desc='Ax DWI') #will update this to fetch more in later datasets if they have more than one plane of dwi images
    if len(dwi_series) != 1: #also upadte this when/if >1 dim
        raise ValueError(
            f"Expected exactly one series named 'Ax DWI'; found {len(dwi_series)}."
        )
    if not db.series(maps[:3]): #chacks resultspath, SUBJ, and 'Maps' are present/valid
        raise ValueError(
            "The Maps study is empty. Run compute_descriptive_maps(DATA, MAPS) "
            "before adc_map so dbdicom can create the ADC series."
        )

    images_by_slice = defaultdict(lambda: defaultdict(list)) #group image arrays by MRI slice selection gradient
    baseline_datasets = {}
    orientation = None #these 3 lines are just starting values changed by images and data
    pixel_shape = None
    missing_b_locations = set() #track unknown b-value slices
    explicit_zero_locations = set() #track slices with b=0

    #read images and check their orientation, ensuring there are no unexpected image rotations 
    for filename in db.files(dwi_series[0]): 
        ds = pydicom.dcmread(filename)
        image_orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        if orientation is None:
            orientation = image_orientation
        elif not np.allclose(image_orientation, orientation, atol=1e-5):
            raise ValueError("Ax DWI images do not share a consistent orientation.")

        #calculate slice position wrt scan direction
        normal = np.cross(orientation[:3], orientation[3:]) 
        normal /= np.linalg.norm(normal)
        slice_location = round(
            float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), normal)),
            3,
        )

        #read b and perform checks for 0 and missing values
        raw_b_value = getattr(ds, 'DiffusionBValue', None)
        if raw_b_value is None:
            b_value = 0.0
            missing_b_locations.add(slice_location)
        else:
            b_value = float(raw_b_value)
            if b_value == 0:
                explicit_zero_locations.add(slice_location)

        #read pixel data and filter for 2D images only
        image = dbdataset.pixel_data(ds)
        if image.ndim != 2:
            raise ValueError("Ax DWI series must contain single-frame 2D images.")
        if pixel_shape is None:
            pixel_shape = image.shape
        elif image.shape != pixel_shape:
            raise ValueError("Ax DWI images do not share a consistent matrix size.")

        ##group images in slice and b-value groups
        images_by_slice[slice_location][b_value].append(image)
        if b_value == 0:
            baseline_datasets[slice_location] = ds

    #check each series containsthese slice and b-value groups
    if not images_by_slice:
        raise ValueError("The 'Ax DWI' series contains no images.")
    slice_locations = sorted(images_by_slice)
    b_values = sorted({b for per_slice in images_by_slice.values() for b in per_slice})
    if 0.0 not in b_values or len(b_values) < 2:
        raise ValueError("Ax DWI must contain a b=0 baseline and at least one nonzero b-value.")
    if missing_b_locations and ( #allows for missing b-value from dicom but not b=0
        explicit_zero_locations
        or missing_b_locations != set(slice_locations)
        or any(len(images_by_slice[location][0.0]) != 1 for location in slice_locations)
    ):
        raise ValueError(
            "Some DWI images lack DiffusionBValue; they can only be treated as "
            "b=0 when there is exactly one such image per slice and no explicit b=0 images."
        )

    #check each slice has the same b-values and repeat counts
    expected_b_values = set(b_values)
    replicate_counts = {}
    for location in slice_locations:
        per_slice = images_by_slice[location]
        if set(per_slice) != expected_b_values:
            raise ValueError("Every DWI slice must contain images for the same b-values.")
        for b_value in b_values:
            count = len(per_slice[b_value])
            if b_value in replicate_counts and replicate_counts[b_value] != count:
                raise ValueError(
                    f"Inconsistent number of images for b={b_value:g} across DWI slices."
                )
            replicate_counts[b_value] = count

    #ensures a 3D/4D array with >1 slice is used
    if len(slice_locations) < 2:
        raise ValueError("At least two DWI slices are required to create a 3D ADC map.")
    slice_spacing = np.diff(slice_locations)
    if not np.allclose(slice_spacing, np.median(slice_spacing), rtol=1e-3, atol=1e-3): #relative and absolute tolerance are 0.1% of the mean spacing
        raise ValueError("Ax DWI slice positions are not regularly spaced.")

    #fit ADC eqn
    signal_slices = []
    for location in slice_locations:
        mean_signals = [
            np.mean(images_by_slice[location][b_value], axis=0)
            for b_value in b_values
        ]
        signal_slices.append(np.stack(mean_signals, axis=-1))
    signal = np.stack(signal_slices, axis=2)
    adc = _fit_adc(signal, b_values)

    #set spatial info
    first_slice = baseline_datasets[slice_locations[0]]
    affine = dbdataset.affine(first_slice) #liunear transformation followed by translation
    if np.dot(affine[:3, 2], normal) < 0: #matrix multiplication of the first 3 rows and 2nd column of the affine matrix with normal matrix
        adc = adc[:, :, ::-1]
        affine[:3, 2] *= -1
    affine[:3, 2] = normal * float(np.median(slice_spacing))

    #write volume and delete any exisiing duplicates
    adc_series = maps + ['ADC']
    db.delete(adc_series, not_exists_ok=True)
    db.write_volume((adc, affine), adc_series, verbose=0)
    print("Finished creating ADC map in native axial DWI space (units: mm^2/s).")

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


def view_adc_notebook(maps):
    """Display the ADC map in napari; run this from a notebook with %gui qt."""
    adc_vol = db.volume(maps + ['ADC'])
    positive_adc = adc_vol.values[np.isfinite(adc_vol.values) & (adc_vol.values > 0)]
    if positive_adc.size == 0:
        raise ValueError("The ADC map contains no positive finite values to display.")

    contrast_max = float(np.percentile(positive_adc, 99))
    voxel_spacing = np.linalg.norm(adc_vol.affine[:3, :3], axis=0)[::-1]
    viewer = napari.Viewer()
    viewer.add_image(
        adc_vol.values.T,
        name='ADC (mm^2/s)',
        scale=voxel_spacing,
        contrast_limits=(0, contrast_max),
        colormap='gray',
    )
    return viewer


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
    from rectify.utils.stats import mask_stats

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
    adc_map(DATAPATH, MAPS)
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
    