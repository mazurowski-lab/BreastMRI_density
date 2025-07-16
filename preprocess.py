import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
import os
import nrrd
import json

def get_instance_number(file_path):
    try:
        ds = pydicom.dcmread(file_path)
        return ds.InstanceNumber
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return float('inf')

def sort_dicom_files_by_instance_number(full_sequence_dir):
    dicom_file_list = [f for f in os.listdir(full_sequence_dir) if f.endswith('.dcm')]
    dicom_file_list = sorted(dicom_file_list, key=lambda f: get_instance_number(os.path.join(full_sequence_dir, f)))
    return dicom_file_list
    
def read_precontrast_mri_1031(
    sequence_dir,
    fpath_mapping_df,
    eval_task
):
    """
    Reads in the precontrast MRI data given a subject ID. 
    This function also aligns the patient orientation so the patient's body
    is in the lower part of image. The slices from the beginning move inferior
    to superior. 


    Parameters
    ----------
    subject_id: str
        Subject_id (e.g. Breast_MRI_001)
    tcia_data_dir: str 
        Path of downloaded database from TCIA
    fpath_mapping_df: pd.DataFrame
        Cleaned mapping DataFrame that can be used to find precontrast MRI
        sequences

    Returns
    -------
    np.Array
        Raw MRI volume data read from all .dcm files
    pydicom.dataset.FileDataset
        Dicom data from final slice read. This is used for obtaining things
        such as pixel spacing, image orientation, etc. 

    """

    full_sequence_dir = sequence_dir
    dicom_file_list = sort_dicom_files_by_instance_number(full_sequence_dir)
    print(len(dicom_file_list))
    if len(dicom_file_list) <= 1:
        print(full_sequence_dir)
    dicom_data_list = []
    # Saving the values of first two image positions
    # This is used to orient inferior to superior
    ISFlip = False
    XYFlip = False
    for i in range(len(dicom_file_list)):
        dicom_data = pydicom.dcmread(os.path.join(full_sequence_dir, dicom_file_list[i]))
        
        #if dicom_data.TemporalPositionIdentifier!= None in dicom_data:
        if 'TemporalPositionIdentifier' in dicom_data:
            time_id = dicom_data.TemporalPositionIdentifier
            if time_id != '1':
                if i ==0:
                    print(full_sequence_dir)
                continue

        if i == 0:
            origin = dicom_data.ImagePositionPatient
            first_image_z = dicom_data.ImagePositionPatient[-1]
            x_spacing, y_spacing = dicom_data.PixelSpacing
            z_spacing = dicom_data.SliceThickness

            direction_x = np.array(dicom_data.ImageOrientationPatient[:3])
            direction_y = np.array(dicom_data.ImageOrientationPatient[3:])
            #direction_z = np.cross(direction_x, direction_y) # possible not suitable for some dicom, corrected below
            direction_z = np.array([0,0,1])
            space_direction_x = direction_x * x_spacing
            space_direction_y = direction_y * y_spacing
            space_direction_z = direction_z * z_spacing

            space_directions = np.column_stack((space_direction_x, space_direction_y, space_direction_z)).tolist()

        elif i == 1:
            second_image_z = dicom_data.ImagePositionPatient[-1]
        dicom_data_list.append(dicom_data.pixel_array)
    # Stack in numpy array
    image_array = np.stack(dicom_data_list, axis=-1)
    xy_size = image_array.shape[0]
    z_size  = image_array.shape[2]
    # Rotate if inferior and superior are flipped
    space_directions[2][2] = second_image_z - first_image_z
    if second_image_z < first_image_z:
        image_array = np.flip(image_array, axis=2)
        ISFlip = True
    if direction_x [0] < 0:
        image_array = np.flip(image_array, axis=1)
    if direction_y[1] < 0:
        image_array = np.flip(image_array, axis=0)

    #np.save('pretest.npy', image_array)
    header = {}

    header['space origin'] = list(origin)  # The origin should be the ImagePositionPatient
    header['space directions'] = space_directions  # Set the space directions based on the above calculations
    header['kinds'] = ['domain', 'domain', 'domain']  # This is a typical setting for NRRD when all axes are spatial
    header['encoding'] = 'gzip'  # Assuming you are not compressing the data
    header['space'] = 'left-posterior-superior'  # The space attribute for LPS coordinate system

    if eval_task == 'full_breast':
        header['Segment0_Name'] = 'breast'
        header['Segment0_ID'] = 'Segment_1'
        header['Segment0_LabelValue'] = '1'
        header['Segment0_Layer'] = '0'
        header['Segment0_Color'] = '0.847059 0.396078 0.309804'

    if eval_task == 'vessel_tissue':
        header['Segment0_Name'] = 'vessel'
        header['Segment1_Name'] = 'tissue'
        header['Segment0_ID'] = 'Segment_1'
        header['Segment1_ID'] = 'Segment_2'
        header['Segment0_LabelValue'] = '1'
        header['Segment1_LabelValue'] = '2'
        header['Segment0_Layer'] = '0'
        header['Segment1_Layer'] = '0'
        header['Segment0_Color'] = '0.847059 0.396078 0.309804'
        header['Segment1_Color'] = '0.901961 0.862745 0.27451'

    return image_array, ISFlip, XYFlip, xy_size, z_size, header, dicom_data


def data_preprocess(df, pre_dir,pre_image_generate,eval_task):
    #df = get_folder_adress_df(raw_dir)
    #df = filter_pre_contrast(df)
    #df = pd.read_csv(r'C:\Users\ll359\OneDrive - Duke University\BreastMRI\Code\ISPY2Process\ISPY2_91Patient_meta.csv')
    #df = df[df['precontrast_dir'].str.contains('100899', na=False)]
    #df['extracted'] = df['subject_id'].str.extract(r'-([0-9]+T[0-9])-', expand=False)
    df['filenames'] = df['Study_Name'] + '.npy'
    df['ISFlip'] = False
    df['XYFlip'] = False
    df['Padded'] = 0
    #df['Predict_dir'] = 0
    df['xy_size'] = 512
    df['z_size'] = 0

    ISFlip_iloc = df.columns.get_loc('ISFlip')
    XYFlip_iloc = df.columns.get_loc('XYFlip')
    Padded_iloc = df.columns.get_loc('Padded')
    xy_size_iloc = df.columns.get_loc('xy_size')
    z_size_iloc = df.columns.get_loc('z_size')
    # Predict_dir_iloc = df.columns.get_loc('Predict_dir')
    df = df.sort_values(by='Study_Name')
    Process_df = df

    patch_size = 96

    ImageList = []
    HeaderList = []
    for i in range(len(df)):
        image_array, ISFlip, XYFlip, xy_size, z_size, header, _ = read_precontrast_mri_1031(df['Sequence_path'].iloc[i], df, eval_task)
        image_array = normalize_image(image_array, min_cutoff=0.001, max_cutoff=0.001)
        image_array = zscore_image(image_array)
        df.iloc[i, ISFlip_iloc] = ISFlip
        df.iloc[i, XYFlip_iloc] = XYFlip
        df.iloc[i, xy_size_iloc] = xy_size
        df.iloc[i, z_size_iloc] = z_size
        # ImageList.append(image_array)
        HeaderList.append(header)

        if pre_image_generate ==True:
        # If you want to pad both sides of the third axis evenly
            padding_size = patch_size - image_array.shape[2]
            if padding_size > 0:
                padding_before = padding_size // 2
                padding_after = padding_size - padding_before
                pad_width = [(0, 0), (0, 0), (padding_before, padding_after)]
                image_array_padded = np.pad(image_array, pad_width=pad_width, mode='constant', constant_values=0)
                # print(image_array.shape)
                df.iloc[i, Padded_iloc] = padding_size
            else:
                image_array_padded = image_array

            image_save_address = os.path.join(pre_dir, df['Study_Name'].iloc[i] + '.npy')
            # df.iloc[i, Predict_dir_iloc] = image_save_address
            np.save(image_save_address, image_array_padded)

    df.to_csv('buffer/Process_df.csv',index=False)
    with open('buffer/HeaderList.json', 'w') as json_file:
        json.dump(HeaderList, json_file, indent=4)



def normalize_image(image_array, min_cutoff = 0.001, max_cutoff = 0.001):
    sorted_array = np.sort(image_array.flatten())

    # Find %ile index and get values
    min_index = int(len(sorted_array) * min_cutoff)
    min_intensity = sorted_array[min_index]

    max_index = int(len(sorted_array) * min_cutoff) * -1
    max_intensity = sorted_array[max_index]

    # Normalize image and cutoff values
    image_array = (image_array - min_intensity) / \
        (max_intensity - min_intensity)
    image_array[image_array < 0.0] = 0.0
    image_array[image_array > 1.0] = 1.0

    return image_array

def zscore_image(image_array):


    image_array = (image_array - np.mean(image_array)) / np.std(image_array)

    return image_array