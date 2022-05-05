import os
import errno
import math
import shutil
import logging
import random
import numpy as np
import sys
import nibabel as nib
from pptx import Presentation
from pptx.util import Inches, Pt
import scipy
from scipy import ndimage as ndi
import skimage.measure
import json
from matplotlib import colors
import matplotlib.pyplot as plt
import numbers
import math
import cv2 as cv
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import traceback


# From: https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def convert_3d_2_flat(in_data_matrix):
    num_voxel = np.prod(in_data_matrix.shape)
    return in_data_matrix.reshape(num_voxel)


def convert_flat_2_3d(in_data_array, im_shape):
    return in_data_array.reshape(im_shape)


def read_file_contents_list(file_name):
    print(f'Reading from file list txt {file_name}', flush=True)
    with open(file_name) as file:
        lines = [line.rstrip('\n') for line in file]
        print(f'Number items: {len(lines)}')
        return lines


def save_file_contents_list(file_name, item_list):
    print(f'Save list to file {file_name}')
    print(f'Number items: {len(item_list)}')
    with open(file_name, 'w') as file:
        for item in item_list:
            file.write(item + '\n')


def get_dice(img1, img2):
    assert img1.shape == img2.shape

    img1 = img1.flatten().astype(float)
    img2 = img2.flatten().astype(float)

    dice_val = 2 * (img1 * img2).sum() / (img1 + img2).sum()

    return dice_val


def get_dice_with_effective_mask(img1, img2, mask):
    assert img1.shape == img2.shape
    assert img1.shape == mask.shape

    mask = mask.flatten().astype(float)
    img1 = img1.flatten().astype(float)
    img2 = img2.flatten().astype(float)

    img1 = img1 * mask
    img2 = img2 * mask

    dice_val = 2 * (img1 * img2).sum() / (img1 + img2).sum()

    return dice_val


def get_range_paral_chunk(total_num_item, chunk_pair):
    num_item_each_chunk = int(math.ceil(float(total_num_item) / float(chunk_pair[1])))
    range_lower = num_item_each_chunk * (chunk_pair[0] - 1)
    # range_upper = num_item_each_chunk * chunk_pair[0] - 1
    range_upper = num_item_each_chunk * chunk_pair[0]
    if range_upper > total_num_item:
        range_upper = total_num_item

    return [range_lower, range_upper]


def get_current_chunk(in_list, chunk_pair):
    chunks_list = get_chunks_list(in_list, chunk_pair[1])
    current_chunk = chunks_list[chunk_pair[0] - 1]
    return current_chunk


def get_chunks_list(in_list, num_chunks):
    return [in_list[i::num_chunks] for i in range(num_chunks)]


def get_nii_filepath_and_filename_list(dataset_root):
    nii_file_path_list = []
    subject_list = os.listdir(dataset_root)
    for i in range(len(subject_list)):
        subj = subject_list[i]
        subj_path = dataset_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            nii_files = os.listdir(sess_path)
            for nii_file in nii_files:
                nii_file_path = sess_path + '/' + nii_file
                nii_file_path_list.append(nii_file_path)
                # nii_file_name_list.append(nii_file)


    return nii_file_path_list


def get_nii_filepath_and_filename_list_flat(dataset_root):
    nii_file_path_list = []
    nii_file_name_list = os.listdir(dataset_root)
    for file_name in nii_file_name_list:
        nii_file_path = os.path.join(dataset_root, file_name)
        nii_file_path_list.append(nii_file_path)

    return nii_file_path_list


def get_nii_filepath_and_filename_list_hierarchy(dataset_root):
    nii_file_path_list = []
    nii_file_name_list = []
    subject_list = os.listdir(dataset_root)
    for i in range(len(subject_list)):
        subj = subject_list[i]
        subj_path = dataset_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            nii_files = os.listdir(sess_path)
            for nii_file in nii_files:
                nii_file_path = sess_path + '/' + nii_file
                nii_file_path_list.append(nii_file_path)
                nii_file_name_list.append(nii_file)

    return nii_file_path_list


def get_dataset_path_list(dataset_root, dataset_type):
    file_path_list = []
    if dataset_type == 'flat':
        file_path_list = get_nii_filepath_and_filename_list_flat(dataset_root)
    elif dataset_type == 'hierarchy':
        file_path_list = get_nii_filepath_and_filename_list(dataset_root)
    else:
        file_path_list = []

    return file_path_list


def resample_spore_nifti(spore_nifti_root, spore_resample_root):
    """
    Resample spore data, using c3d
    :param spore_nifti_root:
    :param spore_resample_root:
    :return:
    """
    spore_nii_file_path_list = []
    spore_nii_file_name_list = []
    subject_list = os.listdir(spore_nifti_root)
    for i in range(len(subject_list)):
        subj = subject_list[i]
        subj_path = spore_nifti_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            nii_files = os.listdir(sess_path)
            for nii_file in nii_files:
                nii_file_path = sess_path + '/' + nii_file
                spore_nii_file_path_list.append(nii_file_path)
                spore_nii_file_name_list.append(nii_file)

    file_count = 1
    for iFile in range(len(spore_nii_file_path_list)):
        # if file_count > 3:
        #     break

        file_path = spore_nii_file_path_list[iFile]
        file_name = spore_nii_file_name_list[iFile]

        output_path = spore_resample_root + '/' + file_name

        print('Read image: ', file_path)

        # command_read_info_str = 'c3d ' + file_path + ' -info-full'
        # os.system(command_read_info_str)

        command_str = 'c3d ' + file_path + ' -resample 256x256x180 -o ' + output_path

        os.system(command_str)

        print('Output file: ', file_name, " {}/{}".format(iFile, len(spore_nii_file_name_list)))
        # command_image_info_str = 'c3d ' + output_path + ' -info-full'
        #
        # os.system(command_image_info_str)

        file_count = file_count + 1


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_image_name_from_path(image_path):
    return os.path.basename(image_path)


def dataset_hierarchy_to_flat(in_folder, out_folder):
    file_path_list = get_nii_filepath_and_filename_list(in_folder)
    for file_idx in range(len(file_path_list)):
        file_path = file_path_list[file_idx]
        print(f'({file_idx}/{len(file_path_list)}), Process image {file_path}.')
        file_name = get_image_name_from_path(file_path)
        out_path = os.path.join(out_folder, file_name)
        if os.path.exists(out_path):
            print(out_path + ' already exist')
        else:
            print('Copy file %s to %s' % (file_path, out_path))
            shutil.copyfile(file_path, out_path)


def get_extension(file_full_path):
    filename, file_extension = os.path.splitext(file_full_path)
    return file_extension


def get_registration_command_non_rigid(registration_method_name,
                                       reg_args_affine,
                                       reg_args_non_rigid,
                                       label_file,
                                       reg_tool_root,
                                       fixed_image,
                                       moving_image,
                                       output_image,
                                       output_mat,
                                       output_trans,
                                       output_affine):
    command_list = []
    actual_output_mat_path = output_mat + '_matrix.txt'

    reg_args = reg_args_non_rigid

    if registration_method_name == 'deformable_deedsBCV':
        linearBCVslow_path = os.path.join(reg_tool_root, 'linearBCVslow')
        deedsBCVslow_path = os.path.join(reg_tool_root, 'deedsBCVslow')
        label_prop_command = ''
        if label_file != '':
            label_prop_command = f'-S {label_file}'
        command_list.append(f'{linearBCVslow_path} -F {fixed_image} -M {moving_image} -O {output_mat}')
        command_list.append(f'{deedsBCVslow_path} {reg_args} -F {fixed_image} -M {moving_image} -O {output_image} -A {actual_output_mat_path} {label_prop_command}')
    elif registration_method_name == 'deformable_deedsBCV_paral':
        linearBCV_path = os.path.join(reg_tool_root, 'linearBCV')
        deedsBCV_path = os.path.join(reg_tool_root, 'deedsBCV')

        label_prop_command = ''
        if label_file != '':
            label_prop_command = f'-S {label_file}'

        command_list.append(f'{linearBCV_path} -F {fixed_image} -M {moving_image} -O {output_mat}')
        command_list.append(f'{deedsBCV_path} {reg_args} -F {fixed_image} -M {moving_image} -O {output_image} -A {actual_output_mat_path} {label_prop_command}')
    elif registration_method_name == 'deformable_niftyreg':
        reg_aladin_path = os.path.join(reg_tool_root, 'reg_aladin')
        reg_f3d_path = os.path.join(reg_tool_root, 'reg_f3d')

        output_mat_real = output_mat.replace('.nii.gz', '.txt')
        output_affine_im = output_affine
        output_non_rigid_trans = output_trans

        command_list.append(
            f'{reg_aladin_path} {reg_args_affine} -ref {fixed_image} -flo {moving_image} -res {output_affine_im} -aff {output_mat_real}'
        )

        # command_list.append(
        #     f'{reg_f3d_path} -voff {reg_args} -maxit 1000 -sx 10 -ref {fixed_image} -flo {moving_image} -aff {output_mat_real} -cpp {output_non_rigid_trans} -res {output_image}'
        # )

        command_list.append(
            f'{reg_f3d_path} {reg_args_non_rigid} -maxit 1000 -ref {fixed_image} -flo {moving_image} -aff {output_mat_real} -cpp {output_non_rigid_trans} -res {output_image}'
        )

    else:
        command_list.append('TODO')

    return command_list


def get_registration_command(registration_method_name, reg_args, label_file, reg_tool_root, fixed_image, moving_image, output_image, output_mat):

    command_list = []
    actual_output_mat_path = output_mat + '_matrix.txt'

    if registration_method_name == 'affine_flirt':
        flirt_path = os.path.join(reg_tool_root, 'flirt')
        command_str = f'{flirt_path} {reg_args} -dof 12 -in {moving_image} -ref {fixed_image} -out {output_image} -omat {output_mat} '
        command_list.append(command_str)
    elif registration_method_name == 'affine_flirt_zhoubing':
        flirt_path = os.path.join(reg_tool_root, 'flirt')
        # 1. Rigid.
        mid_step_rigid_mat = output_mat + "_rigid.txt"
        mid_step_rigid_im = output_mat + "_rigid.nii.gz"
        command_list.append(f'{flirt_path} -v -dof 6 -in {moving_image} -ref {fixed_image} -omat {mid_step_rigid_mat} -out {mid_step_rigid_im} -nosearch')
        # 2. DOF 9 Affine.
        command_list.append(f'{flirt_path} -v -dof 9 -in {moving_image} -ref {fixed_image} -init {mid_step_rigid_mat} -omat {output_mat} -out {output_image} -nosearch')
    elif registration_method_name == 'affine_nifty_reg':
        reg_aladin_path = os.path.join(reg_tool_root, 'reg_aladin')
        output_mat_real = output_mat.replace('.nii.gz', '.txt')
        command_list.append(f'{reg_aladin_path} -ln 5 -ref {fixed_image} -flo {moving_image} -res {output_image} -aff {output_mat_real}')
    elif registration_method_name == 'affine_nifty_reg_mask':
        reg_aladin_path = os.path.join(reg_tool_root, 'reg_aladin')
        output_mat_real = output_mat.replace('.nii.gz', '.txt')
        fixed_image_mask = fixed_image.replace('.nii.gz', '_mask.nii.gz')
        moving_image_mask = moving_image.replace('.nii.gz', '_mask.nii.gz')
        command_list.append(f'{reg_aladin_path} -ln 5 -ref {fixed_image} -rmask {fixed_image_mask} -flo {moving_image} -fmask {moving_image_mask} -res {output_image} -aff {output_mat_real}')
    elif registration_method_name == 'rigid_nifty_reg':
        reg_aladin_path = os.path.join(reg_tool_root, 'reg_aladin')
        output_mat_real = output_mat.replace('.nii.gz', '.txt')
        command_list.append(
            f'{reg_aladin_path} -rigOnly -ln 5 -ref {fixed_image} -flo {moving_image} -res {output_image} -aff {output_mat_real}')
    elif registration_method_name == 'affine_deedsBCV':
        linearBCVslow_path = os.path.join(reg_tool_root, 'linearBCVslow')
        applyLinearBCVfloat_path = os.path.join(reg_tool_root, 'applyLinearBCVfloat')
        command_list.append(f'{linearBCVslow_path} -F {fixed_image} -M {moving_image} -O {output_mat}')
        command_list.append(f'{applyLinearBCVfloat_path} -M {moving_image} -A {actual_output_mat_path} -D {output_image}')
    elif registration_method_name == 'deformable_deedsBCV':
        linearBCVslow_path = os.path.join(reg_tool_root, 'linearBCVslow')
        deedsBCVslow_path = os.path.join(reg_tool_root, 'deedsBCVslow')

        label_prop_command = ''
        if label_file != '':
            label_prop_command = f'-S {label_file}'

        command_list.append(f'{linearBCVslow_path} -F {fixed_image} -M {moving_image} -O {output_mat}')
        command_list.append(f'{deedsBCVslow_path} {reg_args} -F {fixed_image} -M {moving_image} -O {output_image} -A {actual_output_mat_path} {label_prop_command}')
    elif registration_method_name == 'deformable_deedsBCV_paral':
        linearBCV_path = os.path.join(reg_tool_root, 'linearBCV')
        deedsBCV_path = os.path.join(reg_tool_root, 'deedsBCV')

        label_prop_command = ''
        if label_file != '':
            label_prop_command = f'-S {label_file}'

        command_list.append(f'{linearBCV_path} -F {fixed_image} -M {moving_image} -O {output_mat}')
        command_list.append(f'{deedsBCV_path} {reg_args} -F {fixed_image} -M {moving_image} -O {output_image} -A {actual_output_mat_path} {label_prop_command}')
    elif registration_method_name == 'deformable_niftyreg':
        reg_aladin_path = os.path.join(reg_tool_root, 'reg_aladin')
        reg_f3d_path = os.path.join(reg_tool_root, 'reg_f3d')

        output_mat_real = output_mat.replace('.nii.gz', '.txt')
        output_affine_im = output_image.replace('.nii.gz', '_affine.nii.gz')
        output_non_rigid_trans = output_image.replace('.nii.gz', '_non_rigid_trans.nii.gz')

        command_list.append(
            f'{reg_aladin_path} -ln 5 -omp 32 -ref {fixed_image} -flo {moving_image} -res {output_affine_im} -aff {output_mat_real}'
        )

        command_list.append(
            f'{reg_f3d_path} -ln 5 -omp 32 -maxit 1000 {reg_args} -ref {fixed_image} -flo {moving_image} -aff {output_mat_real} -cpp {output_non_rigid_trans} -res {output_image}'
        )

    else:
        command_list.append('TODO')

    return command_list


def get_interpolation_command(interp_type_name, bash_config, src_root, moving_image):

    command_list = []
    file_name = moving_image
    real_mat_name = file_name.replace('nii.gz', 'txt')

    bash_script_path = ''
    if interp_type_name == 'clipped_ori':
        bash_script_path = os.path.join(src_root, 'tools/interp_clipped_roi.sh')
    elif interp_type_name == 'full_ori':
        bash_script_path = os.path.join(src_root, 'tools/interp_full_ori.sh')
    elif interp_type_name == 'roi_lung_mask':
        bash_script_path = os.path.join(src_root, 'tools/interp_ori_lung_mask.sh')

    command_list.append(f'{bash_script_path} {bash_config} {file_name} {real_mat_name}')

    return command_list


loggers = {}


def get_logger(name, log_file=None, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file is not None:
            fileHandler = logging.FileHandler(log_file)
            fileHandler.setFormatter(formatter)
            logger.addHandler(fileHandler)

        loggers[name] = logger

        return logger


logger = get_logger('Utils')


def check_standard_orientation(nii_path):
    img_nib = nib.load(nii_path)
    aff_mat = img_nib.affine
    if aff_mat[0][0] < 0 and aff_mat[1][1] > 0 and aff_mat[2][2] > 0 and aff_mat[3][3] > 0:
        return 1
    else:
        return 0


def convert_std_orientation(in_nii_path, out_nii_path):
    fslreorient2std_str = '/usr/local/fsl/bin/fslreorient2std'
    cmd_str = f'{fslreorient2std_str} {in_nii_path} {out_nii_path}'
    print(cmd_str)
    os.system(cmd_str)


def save_pptx(png_file_list, pptx_path):
    prs = Presentation()
    layout_image = prs.slide_layouts[6]
    left = top = Inches(0)
    height = Inches(5.5)
    font_size = Pt(10)
    left_text = Inches(1)
    top_text = Inches(6.6)

    for png_file_path in png_file_list:
        print(f'Add {png_file_path}')
        slide = prs.slides.add_slide(layout_image)
        slide.shapes.add_picture(png_file_path, left, top, height=height)

        txBox = slide.shapes.add_textbox(left_text, top_text, Inches(1), Inches(1))
        tf = txBox.text_frame
        tf.text = os.path.basename(png_file_path.replace('.png', ''))
        # tf.font.size = font_size

    print(f'Save to {pptx_path}')
    prs.save(pptx_path)


def get_lung_level_boundary(lung_mask):
    # Get the border information
    size_x, size_y, size_z = lung_mask.shape

    z_label_flag_arr = np.zeros((size_z,), dtype=int)
    for idx_slice in range(size_z):
        axial_slice = lung_mask[:, :, idx_slice]
        label_sum = np.sum(axial_slice)
        if label_sum > 10:
            z_label_flag_arr[idx_slice] = 1

    pos_idx_list = np.where(z_label_flag_arr == 1)
    low_bd = np.min(np.array(pos_idx_list))
    up_bd = np.max(np.array(pos_idx_list))

    range_z = up_bd - low_bd
    cut_range = int(range_z * 1.1)
    ext_size = int((cut_range - range_z) / 2)
    up_bd += ext_size
    low_bd -= ext_size
    up_bd = np.min([up_bd, size_z - 1])
    low_bd = np.max([low_bd, 0])

    return low_bd, up_bd, size_x, size_y


def extract_lung_region(ct_path, lung_mask_path, out_path):
    """
    Extract the lung region by crop the z dimension.
    :param nii_path:
    :return:
    """
    img_obj = nib.load(ct_path)
    print(f'Shape of input nii:')
    print(img_obj.get_data().shape)

    print('Get lung seg:')
    lung_mask = nib.load(lung_mask_path).get_data()
    print('Done')

    # Get the border information
    low_bd, up_bd, size_x, size_y = get_lung_level_boundary(lung_mask)

    range_z = up_bd - low_bd
    c3d_str = '/home/local/VANDERBILT/xuk9/bin/c3d'
    cmd_str = f'{c3d_str} {ct_path} -region 0x0x{low_bd}vox {size_x}x{size_y}x{range_z}vox -o {out_path}'
    print(cmd_str)
    os.system(cmd_str)


def get_lung_extract_boundary_3D(lung_mask):
    size_list = list(lung_mask.shape)
    label_flag_views = []
    label_flag_views.append(np.sum(lung_mask, axis=(1, 2)))
    label_flag_views.append(np.sum(lung_mask, axis=(0, 2)))
    label_flag_views.append(np.sum(lung_mask, axis=(0, 1)))

    low_bd_list = []
    up_bd_list = []
    for view_idx, view_str in enumerate(['X', 'Y', 'X']):
        pos_idx_list = np.where(label_flag_views[view_idx] > 10)
        low_bd = np.min(np.array(pos_idx_list))
        up_bd = np.max(np.array(pos_idx_list))
        range_view = up_bd - low_bd
        cut_range = int(range_view * 1.2)   # At the training stage, use cut ratio of 1.1
        ext_size = int((cut_range - range_view) / 2)
        up_bd += ext_size
        low_bd -= ext_size
        up_bd = np.min([up_bd, size_list[view_idx] - 1])
        low_bd = np.max([low_bd, 0])

        low_bd_list.append(low_bd)
        up_bd_list.append(up_bd)

    return low_bd_list, up_bd_list, size_list


def extract_lung_region_3D(ct_path, lung_mask_path, out_path):
    """
    Extract 3D ROI using lung mask
    :param ct_path:
    :param lung_mask_path:
    :param out_path:
    :return:
    """
    img_obj = nib.load(ct_path)
    print(f'Shape of input nii:')
    print(img_obj.get_data().shape)

    print('Get lung seg:')
    lung_mask = nib.load(lung_mask_path).get_data()
    print('Done')

    low_bd_list, up_bd_list, size_list = get_lung_extract_boundary_3D(lung_mask)
    range_list = [up_bd - low_bd for low_bd, up_bd in zip(low_bd_list, up_bd_list)]

    c3d_str = '/home/local/VANDERBILT/xuk9/bin/c3d'
    start_loc_str = f'{low_bd_list[0]}x{low_bd_list[1]}x{low_bd_list[2]}vox'
    size_str = f'{range_list[0]}x{range_list[1]}x{range_list[2]}vox'
    cmd_str = f'{c3d_str} {ct_path} -region {start_loc_str} {size_str} -o {out_path}'
    print(cmd_str)
    os.system(cmd_str)


def smooth_image(in_image, std=1):
    out_image_data = scipy.ndimage.gaussian_filter(in_image.astype(np.float32), std)
    return out_image_data


def create_body_mask(in_img, out_mask):
    rBody = 2

    print(f'Get body mask of image {in_img}')

    image_nb = nib.load(in_img)
    image_np = np.array(image_nb.dataobj)

    BODY = (image_np>=-500)# & (I<=win_max)
    print(f'{np.sum(BODY)} of {np.size(BODY)} voxels masked.')
    if np.sum(BODY)==0:
      raise ValueError('BODY could not be extracted!')

    # Find largest connected component in 3D
    struct = np.ones((3,3,3),dtype=np.bool)
    BODY = ndi.binary_erosion(BODY,structure=struct,iterations=rBody)

    BODY_labels = skimage.measure.label(np.asarray(BODY, dtype=np.int))

    props = skimage.measure.regionprops(BODY_labels)
    areas = []
    for prop in props:
      areas.append(prop.area)
    print(f' -> {len(areas)} areas found.')
    # only keep largest, dilate again and fill holes
    BODY = ndi.binary_dilation(BODY_labels==(np.argmax(areas)+1),structure=struct,iterations=rBody)
    # Fill holes slice-wise
    for z in range(0,BODY.shape[2]):
      BODY[:,:,z] = ndi.binary_fill_holes(BODY[:,:,z])

    new_image = nib.Nifti1Image(BODY.astype(np.int8), header=image_nb.header, affine=image_nb.affine)
    nib.save(new_image,out_mask)
    print(f'Generated body_mask segs in Abwall {out_mask}')


def merge_list_array(in_list_array):
    return [item for sublist in in_list_array for item in sublist]


def save_json(data, out_path):
    out_file = open(out_path, 'w')
    logger.info(f'Save to {out_path}')
    json.dump(data, out_file)
    out_file.close()


def load_json(in_path):
    with open(in_path) as json_file:
        logger.info(f'Load {in_path}')
        load_data = json.load(json_file)
        return load_data


def plot_compare_overlay_ct(image, mask, out_png):
    show_row_image = np.concatenate([
      image, image
    ], axis=1)

    show_row_mask = np.concatenate([
      np.zeros(image.shape), mask
    ], axis=1).astype(float)

    show_row_mask[show_row_mask == 0] = np.nan
    show_row_mask[show_row_mask == -1] = np.nan

    cmap = colors.ListedColormap(['#00FFFF', '#FF0000'])
    boundaries = [0.5, 1.5, 2.5]

    fig, ax = plt.subplots()
    plt.axis('off')

    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    ax.imshow(
      show_row_image,
      interpolation='bilinear',
      cmap='gray',
      # norm=colors.Normalize(vmin=-1, vmax=1),
      norm=colors.Normalize(vmin=-1200, vmax=600),
      alpha=0.8)
    ax.imshow(
      show_row_mask,
      interpolation='none',
      cmap=cmap,
      norm=norm,
      alpha=0.5
    )

    print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def plot_compare_gt_pred(image, pred, mask, out_png):
    show_row_image = np.concatenate([
      image, image
    ], axis=1)

    show_row_mask = np.concatenate([
      mask, pred
    ], axis=1).astype(float)
    show_row_mask[show_row_mask == 0] = np.nan
    show_row_mask[show_row_mask == -1] = np.nan

    cmap = colors.ListedColormap(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#ffff00'])
    boundaries = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    fig, ax = plt.subplots()
    plt.axis('off')

    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    ax.imshow(
      show_row_image,
      interpolation='bilinear',
      cmap='gray',
      # norm=colors.Normalize(vmin=-1, vmax=1),
      norm=colors.Normalize(vmin=79, vmax=255),
      alpha=0.8)
    ax.imshow(
      show_row_mask,
      interpolation='none',
      cmap=cmap,
      norm=norm,
      alpha=0.5
    )

    print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def plot_combined_compare_pred(case_filename, image_dir, pred_dir, out_png):
    level_list = ['T5', 'T8', 'T10']

    image_list = []
    pred_list = []
    for level in level_list:
        image_path = os.path.join(image_dir, case_filename.replace('.nii.gz', f'_{level}.png'))
        pred_path = os.path.join(pred_dir, case_filename.replace('.nii.gz', f'_{level}.png'))
        if not (os.path.exists(image_path) and os.path.exists(pred_path)):
            continue

        # image = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)[:, :, 0]
        # pred = cv2.imread(pred_path, flags=cv2.IMREAD_GRAYSCALE)[:, :, 0]

        image = cv.imread(image_path)
        pred = cv.imread(pred_path)
        image = image[:, :, 0]
        pred = pred[:, :, 0]

        image_list.append(image)
        pred_list.append(pred)

    if (len(image_list) < 3) or (len(pred_list) < 3):
        logger.info(f'Number of valid number of levels < 3')
        return

    show_row_image = np.concatenate(image_list, axis=1)
    show_row_mask = np.concatenate(pred_list, axis=1)

    show_row_image = np.concatenate([show_row_image, show_row_image], axis=0)
    show_row_mask = np.concatenate([np.zeros(show_row_mask.shape), show_row_mask], axis=0)

    show_row_mask[show_row_mask == 0] = np.nan
    show_row_mask[show_row_mask == -1] = np.nan

    cmap = colors.ListedColormap(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#ffff00'])
    boundaries = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    fig, ax = plt.subplots()
    plt.axis('off')

    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    ax.imshow(
      show_row_image,
      interpolation='bilinear',
      cmap='gray',
      # norm=colors.Normalize(vmin=-1, vmax=1),
      norm=colors.Normalize(vmin=79, vmax=255),
      alpha=0.8)
    ax.imshow(
      show_row_mask,
      interpolation='none',
      cmap=cmap,
      norm=norm,
      alpha=0.5
    )

    print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def clip_image(image_data, clip_plane, iloc):
    clip = None
    if clip_plane == 'sagittal':
        clip = image_data[iloc, :, :]
        clip = np.flip(clip, 0)
        clip = np.rot90(clip)
    elif clip_plane == 'coronal':
        clip = image_data[:, iloc, :]
        clip = np.rot90(clip)
    elif clip_plane == 'axial':
        clip = image_data[:, :, iloc]
        clip = np.rot90(clip)
    else:
        raise NotImplementedError

    return clip


# def is_number(var):
#     return isinstance(var, int) or isinstance(var, float)


def bootstrap_ci(stats_func, input_arr, alpha=0.95, n_size_ratio=1.0, n_iterations=100):
    random.seed(15)
    n_size = round(len(input_arr) * n_size_ratio)
    stats = []
    for i in range(n_iterations):
        sample_arr = random.choices(input_arr, k=n_size)
        stat = stats_func(sample_arr)
        stats.append(stat)

    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    # upper = min(1.0, np.percentile(stats, p))
    upper = np.percentile(stats, p)

    return lower, upper


def is_number(var):
    return isinstance(var, numbers.Number) and (not math.isnan(var))


def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class AverageMeter:
    """
    AverageMeter implements a class which can be used to track a metric over the entire training process.
    (see https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all class variables to default values
        """
        self.val = 0
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates class variables with new value and weight
        """
        self.val = val
        self.vals.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        """
        Implements format method for printing of current AverageMeter state
        """
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )


class AverageMeterSet:
    """
    AverageMeterSet implements a class which can be used to track a set of metrics over the entire training process
    based on AverageMeters (Source: https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=""):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix="/avg"):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix="/sum"):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix="/count"):
        return {name + postfix: meter.count for name, meter in self.meters.items()}