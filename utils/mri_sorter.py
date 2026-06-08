from .orthanc_utils import *
from .classifier_utils import *
import pandas as pd
import numpy as np
import re
import streamlit as st


class MRI_Sorter:
    def __init__(self, study):
        all_series = list(study.series_dict.values())
        already_processed = [s for s in all_series if s.dl_orthanc_id is not None or s.roundel_orthanc_id is not None]
        already_typed = [s for s in all_series if s.series_type is not None and s.dl_orthanc_id is None and s.roundel_orthanc_id is None]
        series_list = [s for s in all_series if s.dl_orthanc_id is None and s.roundel_orthanc_id is None and s.series_type is None]
        print(f"[SORTER] Study has {len(all_series)} series total: "
              f"{len(already_processed)} skipped (already processed), "
              f"{len(already_typed)} skipped (already typed), "
              f"{len(series_list)} to classify")
        sort_df = pd.DataFrame()

        if series_list:
            self.dicom_info = pd.concat([self.extract_series_df(s) for s in series_list])

            dicom_info_3D = self.dicom_info.loc[self.dicom_info['Dimension'] == 3]
            sort_dict_3D = self.sort_3D(dicom_info_3D)

            dicom_info_2D = self.dicom_info.loc[self.dicom_info['Dimension'] == 2]
            sort_dict_2D = self.sort_2D(dicom_info_2D) 

            sort_dict = {**sort_dict_2D, **sort_dict_3D}
            sort_df = pd.DataFrame(sort_dict).T
        self.sort_df = sort_df


    def orientation_id(self, iop):
        """Get ImageOrientationPatient of dicom object.

        Returns one of
            Transverse
            Coronal
            Sagittal
            NA  (if ImageOrientationPatient not available)
        """
        iop_rounded = [round(x,1) for x in iop]
        plane_cross = np.cross(iop_rounded[0:3], iop_rounded[3:6])
        plane = [abs(x) for x in plane_cross]

        if plane[0] == 1:
            return 'Sagittal'
        elif plane[1] == 1:
            return 'Coronal'
        elif plane[2] == 1:
            return 'Axial'
        else:
            return np.nan
    
    def classify_series(self, series_df, series_type, dimension):
        raw_desc = series_df.iloc[0]["SeriesDescription"]
        description = raw_desc.lower()
        keywords = {"SAX": ["sax", "short"], "4CH": ["4ch"], "Other": ["2ch", "3ch", "r2ch", "lvot", "rvot"]}
        for label, kws in keywords.items():
            if any(k in description for k in kws):
                print(f"[CLASSIFY] '{raw_desc}' → keyword match: {label}")
                return label

        earliest_frame_ids = (
            series_df.sort_values('InstanceNumber')
            .groupby('SeriesUID')['ID']
            .first()
        )
        slices = [fetch_orthanc_dicom(iid).pixel_array for iid in earliest_frame_ids]
        image_3d = np.stack(slices)  # (S, H, W)
        print(f"[DEBUG CLASSIFER] - image_3d shape: {image_3d.shape}")

        # Apply classifier model
        probs_list = {}
        for s in range(image_3d.shape[0]):
            image_2d = image_3d[s, ...]  # (H, W)
            with torch.no_grad():
                logits = model(preprocess_resnet_classifier(image_2d).to(device))
                probs = torch.softmax(logits, dim=1)
                probs_list[s] = probs.squeeze(0).cpu()

        # Add probabilities over all slices and average with gaussian weighting
        all_probs = torch.stack(list(probs_list.values()), dim=0)  # (S, num_classes)
        n_slices = all_probs.shape[0]
        mid = (n_slices - 1) / 2.0
        sigma = n_slices / 4.0
        indices = torch.arange(n_slices, dtype=torch.float32)
        weights = torch.exp(-0.5 * ((indices - mid) / sigma) ** 2)  # Gaussian centred on middle slice
        weights = weights / weights.sum()  # normalise so weights sum to 1
        # print(f"Gaussian weights for each slice: {weights}")
        weighted_probs = (all_probs * weights.unsqueeze(1)).sum(dim=0)  # (num_classes,)
        # print(f"Weighted average probs across slices: {weighted_probs}")
        pred = torch.argmax(weighted_probs).item()

        # Get corresponding label
        label = label_dict[pred]
        prob_str = ", ".join(f"{label_dict[i]} (p={weighted_probs[i]:.2f})" for i in range(len(weighted_probs)))
        print(f"[CLASSIFY] '{raw_desc}' → no keyword match, CNN: {prob_str} → {label}")
        return label
                
    def update_sort_dict(self, sort_dict, dimension, all_series_list, series_type, flow_flag, cine_flag, stack_flag):
        for group_tag, all_series_df in enumerate(all_series_list):
            if series_type == 'Cine Stack':
                orientation =  self.classify_series(all_series_df, series_type, dimension)
            else:
                orientation =  self.orientation_id(all_series_df.iloc[0].ImageOrientationPatient)
            for series_uid, series_df in all_series_df.groupby('SeriesUID'):
                sort_dict[series_uid].update({
                    'Dimension': dimension,
                    'Flow': flow_flag,
                    'Cine': cine_flag,
                    'Stack':stack_flag,
                    'Orientation': orientation,
                    'Type': series_type,
                    'Slices': series_df['SliceLocation'].nunique(),
                    'Frames': int(len(series_df) / series_df['SliceLocation'].nunique()),
                    'Group': f'{series_type}:{group_tag:02}'
                })
        return sort_dict

    def extract_series_df(self, series):
        desc = getattr(series, "series_description", series.orthanc_series_id)
        instance_list = fetch_orthanc_instances_for_series(series.orthanc_series_id)

        if not instance_list:
            print(f"[EXTRACT] '{desc}' → skipped: no instances")
            return pd.DataFrame()

        # read first DICOM for shared series-level fields and RR
        ds0 = fetch_orthanc_dicom(instance_list[0]['ID'])
        if not ("PixelData" in ds0 and "ImageOrientationPatient" in ds0):
            print(f"[EXTRACT] '{desc}' → skipped: no PixelData/IOP")
            return pd.DataFrame()

        image_shape = (ds0.Rows, ds0.Columns)
        dimension = int(getattr(ds0, "MRAcquisitionType", "0").replace("D", "")) if hasattr(ds0, "MRAcquisitionType") else None
        print(f"[EXTRACT] '{desc}' → Dimension={dimension}, N={len(instance_list)}")
        series_fields = {
            "SeriesUID": series.orthanc_series_id,
            "SeriesDescription": getattr(ds0, "SeriesDescription", None),
            "PixelSpacing": getattr(ds0, "PixelSpacing")[0],
            "SliceThickness": getattr(ds0, "SpacingBetweenSlices", getattr(ds0, "SliceThickness", np.nan)),
            "Dimension": int(getattr(ds0, "MRAcquisitionType", "0").replace("D", "")) if hasattr(ds0, "MRAcquisitionType") else np.nan,
            "N_timesteps": int(getattr(ds0, "CardiacNumberOfImages", 0)),
            "ImageShape": image_shape,
            "venc": 0,
        }

        # compute RR only from first DICOM
        try:
            rr_ni = float(ds0.NominalInterval)
        except Exception:
            rr_ni = 0.0
        try:
            rr_hr = 60000.0 / float(ds0.HeartRate)
        except Exception:
            rr_hr = 0.0
        rr = max(rr_ni, rr_hr)
        series_fields["rr"] = rr if 100 < rr < 3000 else 0.0

        # build records using only MainDicomTags
        records = [
            {
                **series_fields,
                "ID": inst['ID'],
                "ImageOrientationPatient": tuple(float(x) for x in inst['MainDicomTags']['ImageOrientationPatient'].split('\\')),
                "ImagePositionPatient": tuple(float(x) for x in inst['MainDicomTags']['ImagePositionPatient'].split('\\')),
                "SliceLocation": float(inst['MainDicomTags']['ImagePositionPatient'].split('\\')[2]),
                "InstanceNumber": int(inst['MainDicomTags']['InstanceNumber'])
            }
            for inst in instance_list
        ]

        return pd.DataFrame(records).sort_values("InstanceNumber")
    
    def _is_contiguous_slicelocation(self, slice_locations, tol=0.15):
        """
        Check whether SliceLocation values are approximately evenly spaced.
        """
        vals = np.sort(slice_locations.dropna().unique())

        if len(vals) < 3:
            return False

        diffs = np.diff(vals)

        mean_diff = np.mean(diffs)
        if mean_diff == 0:
            return False

        rel_var = np.std(diffs) / mean_diff

        return rel_var < tol

    def sort_cine_stacks(self, cine_non_flow_list):
        """
        Sort the 2D DICOMs into cine stacks and update the sort dictionary.
        """
        cine_non_flow_df = pd.concat(cine_non_flow_list)
        cine_stack_list = []

        for unique_or in cine_non_flow_df['ImageOrientationPatient'].dropna().unique():
            unique_df = cine_non_flow_df[
                cine_non_flow_df['ImageOrientationPatient'] == unique_or
            ].reset_index()

            unique_df = unique_df.set_index(
                ['PixelSpacing', 'SliceThickness', 'N_timesteps']
            ).sort_index()

            for unique_idx, stack_df in unique_df.groupby(level=[0, 1, 2]):
                for unique_imshape, shape_group in stack_df.groupby('ImageShape'):

                    shape_group = shape_group.reset_index()

                    if len(shape_group) > 1:
                        separated = False

                        for uni_series, series_df in shape_group.groupby('SeriesUID'):
                            n_frames = len(series_df)
                            n_slices = series_df['SliceLocation'].nunique()
                            desc_val = series_df.iloc[0].get('SeriesDescription', uni_series)
                            contiguous = self._is_contiguous_slicelocation(series_df['SliceLocation'])
                            if n_frames > 50 and n_slices > 1:
                                print(f"[STACK] '{desc_val}' n_frames={n_frames} n_slices={n_slices} contiguous={contiguous} → {'accepted' if contiguous else 'rejected'}")
                                if contiguous:
                                    cine_stack_list.append(series_df)
                                    separated = True
                            else:
                                print(f"[STACK] '{desc_val}' n_frames={n_frames} n_slices={n_slices} → rejected (too few frames/slices)")

                        if (
                            not separated
                            and shape_group['SliceLocation'].nunique() > 1
                            and self._is_contiguous_slicelocation(shape_group['SliceLocation'])
                        ):
                            cine_stack_list.append(shape_group)

        if cine_stack_list:
            cine_stack_seriesuids = pd.concat(cine_stack_list)['SeriesUID'].unique()
            cine_single_list = [
                df for df in cine_non_flow_list
                if df['SeriesUID'].iloc[0] not in cine_stack_seriesuids
            ]
        else:
            cine_single_list = cine_non_flow_list

        return cine_stack_list, cine_single_list

    def get_venc(self, cine_single_list):
        for series_df in cine_single_list:
            max_venc = 0

            idxs = (0, len(series_df)//2, len(series_df)-1)
            for i in idxs:
                ds = fetch_orthanc_dicom(series_df.iloc[i]['ID'])
                manufacturer = (getattr(ds, 'Manufacturer', '') or '').lower()

                venc = 0
                if 'siemens' in manufacturer:
                    val = getattr(ds.get((0x0051, 0x1014)), 'value', None)
                    if val:
                        nums = re.findall(r'\d+', str(val))
                        if nums:
                            venc = float(max(map(int, nums)))
                    if not venc:
                        val = getattr(ds.get((0x0018, 0x0024)), 'value', '')
                        m = re.search(r'v(\d+)in', str(val))
                        if m:
                            venc = float(m.group(1))

                elif 'ge' in manufacturer:
                    val = getattr(ds.get((0x0019, 0x10cc)), 'value', None)
                    if val:
                        venc = val / 10

                elif 'philips' in manufacturer:
                    seq = getattr(ds, 'RealWorldValueMappingSequence', None)
                    if seq:
                        intercept = getattr(seq[0], 'RealWorldValueIntercept', None)
                        if intercept is not None:
                            venc = abs(intercept)
                    if not venc:
                        intercept = getattr(ds, 'RescaleIntercept', None)
                        if intercept is not None:
                            venc = abs(intercept)

                max_venc = max(max_venc, venc)

            series_df['venc'] = max_venc

        return cine_single_list
    

    def split_by_similar_triggertime(self, cine_stack_list, cv_threshold=0.1):
        new_cine_stack_list = []
        ss_mh_list = []
        cine_stack_multi_heartbeat = []

        min_slices = 5
        min_timesteps = 10
        max_timesteps = 50

        for series_df in cine_stack_list:
            n_images = len(series_df)
            uid = series_df['SeriesUID'].iloc[0]
            desc_val = series_df.iloc[0].get('SeriesDescription', uid)

            n_slices = series_df.SliceLocation.nunique()

            first_slice = series_df.SliceLocation.values[0]
            n_times = (
                series_df.loc[series_df["SliceLocation"] == first_slice]
                .InstanceNumber.nunique()
            )

            # route oversized stacks directly
            if n_times > max_timesteps:
                print(f"[SPLIT] '{desc_val}' n_times={n_times} n_slices={n_slices} → multi-heartbeat (n_times>{max_timesteps})")
                cine_stack_multi_heartbeat.append(series_df)
                continue


            # sample trigger times
            idxs = (0, len(series_df) // 2, len(series_df) - 1)
            triggertimes = []

            for i in idxs:
                ds = fetch_orthanc_dicom(series_df.iloc[i]["ID"])
                tt = getattr(ds, "TriggerTime", 0) or 0
                triggertimes.append(float(tt))

            triggertimes = np.array(triggertimes)

            mean = triggertimes.mean()
            std = triggertimes.std()
            cv = std / mean if mean else 0

            is_constant = np.all(triggertimes == triggertimes[0])
            is_valid_size = n_slices > min_slices and n_times > min_timesteps

            if is_constant:
                bucket = "Cine Stack" if is_valid_size else "SS_MH (too small)"
                print(f"[SPLIT] '{desc_val}' n_times={n_times} n_slices={n_slices} triggertimes=constant is_valid_size={is_valid_size} → {bucket}")
                (new_cine_stack_list if is_valid_size else ss_mh_list).append(series_df)
                continue

            if cv < cv_threshold:
                print(f"[SPLIT] '{desc_val}' n_times={n_times} n_slices={n_slices} cv={cv:.3f} → SS_MH (low cv)")
                ss_mh_list.append(series_df)
            elif is_valid_size:
                print(f"[SPLIT] '{desc_val}' n_times={n_times} n_slices={n_slices} cv={cv:.3f} is_valid_size={is_valid_size} → Cine Stack")
                new_cine_stack_list.append(series_df)
            else:
                print(f"[SPLIT] '{desc_val}' n_times={n_times} n_slices={n_slices} cv={cv:.3f} is_valid_size={is_valid_size} → dropped")

        return new_cine_stack_list, ss_mh_list, cine_stack_multi_heartbeat
        
    def sort_2D(self, dicom_info_2D):

        sort_dict_2D = {}

        for series_uid, series_df in dicom_info_2D.groupby('SeriesUID'):
            # Initialize the dictionary for each series
            sort_dict_2D[series_uid] = {
                'Description': series_df.iloc[0]['SeriesDescription'],
                'N': len(series_df),
                'Dimension': 2
            }

        # Initialize lists to store results
        series_df_list = []

        # Process unique ImageOrientationPatients
        for unique_or in dicom_info_2D['ImageOrientationPatient'].dropna().unique():
            # Filter and sort DataFrame
            unique_df = dicom_info_2D[dicom_info_2D['ImageOrientationPatient'] == unique_or]
            unique_df = unique_df.set_index(['PixelSpacing', 'SliceThickness']).sort_index()

            # Group by unique indices (PixelSpacing, SliceThickness)
            for unique_idx, series_df in unique_df.groupby(level=[0, 1]):
                # Group by image shape
                for unique_imshape, shape_group in series_df.groupby('ImageShape'):
                    # Filter for series with more than one entry
                    series_df_list.append(series_df)

        # Classify static and cine based on instance number for a slice location
        static_df_list = [
                    df
                    for df in series_df_list
                    if df.loc[df['SliceLocation'] == df['SliceLocation'].unique()[0]]['InstanceNumber'].nunique() == 1
                ]


        # Classify static data based on slicelocation
        static_single_list = [df for df in static_df_list if df['SliceLocation'].nunique() == 1]
        static_stack_list = [df for df in static_df_list if df['SliceLocation'].nunique() > 1]

        sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, static_single_list, 'Static Single', flow_flag=0, cine_flag=0, stack_flag = 0)
        sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, static_stack_list, 'Static Stack', flow_flag=0, cine_flag=0, stack_flag = 1)
                
        cine_df_list = [
                    df
                    for df in series_df_list
                    if df.loc[df['SliceLocation'] == df['SliceLocation'].unique()[0]]['InstanceNumber'].nunique() > 1
                ]
        
        if cine_df_list:
            cine_df_list = self.get_venc(cine_df_list)
            cine_non_flow_list =  [df for df in cine_df_list if not (df['venc'] > 0).any()]

            cine_stack_list, cine_single_list = self.sort_cine_stacks(cine_non_flow_list) # get cine stacks

            cine_2d_flow_list = [df for df in cine_single_list if (df['venc'] > 0).any()]

            cine_stack_list, ss_mh_list, cine_stack_multi_heartbeat = self.split_by_similar_triggertime(cine_stack_list)

            sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, cine_single_list, 'Cine Single', flow_flag=0, cine_flag=0, stack_flag = 0)
            sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, cine_stack_list, 'Cine Stack', flow_flag=0, cine_flag=0, stack_flag = 1)
            sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, cine_stack_multi_heartbeat, 'Cine Stack Multi-Heartbeat', flow_flag=0, cine_flag=0, stack_flag = 1)
            sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, ss_mh_list, 'Single-Short Multi-Heartbeat', flow_flag=0, cine_flag=0, stack_flag = 0)
            sort_dict_2D = self.update_sort_dict(sort_dict_2D, 2, cine_2d_flow_list, '2D Flow', flow_flag=1, cine_flag=1, stack_flag = 0)

        return sort_dict_2D
    
    def sort_3D(self, dicom_info_3D):
        '''
        sort the 3D dicoms
        '''
        sort_dict_3D = {}
        for series_uid, series_df in dicom_info_3D.groupby('SeriesUID'):
            sort_dict_3D[series_uid] = {
                'Description': series_df.iloc[0]['SeriesDescription'],
                'N': len(series_df),
                'Dimension': 3
            }

        # Initialize lists to store results
        stack_df_list = []

        # Process unique orientations
        for unique_or in dicom_info_3D['ImageOrientationPatient'].dropna().unique():
            # Filter and sort DataFrame
            unique_df = dicom_info_3D[dicom_info_3D['ImageOrientationPatient'] == unique_or]
            unique_df = unique_df.set_index(['PixelSpacing', 'SliceThickness']).sort_index()

            # Group by unique indices (pixelspacing, thickness)
            for unique_idx, stack_df in unique_df.groupby(level=[0, 1]):
                # Group by image shape
                for unique_imshape, shape_group in stack_df.groupby('ImageShape'):
                    if len(shape_group) > 1:
                        stack_df_list.append(shape_group)

        # Classify single-phase and multi-phase datasets
        stack_df_list = self.get_venc(stack_df_list)

        flow_4d = [df for df in stack_df_list if (df['venc'] > 0).any()]
        single_phase_list = [df for df in stack_df_list if df['ImagePositionPatient'].nunique() == len(df) and not (df['venc'] != 0).any()]
        multi_phase_list = [df for df in stack_df_list if df['ImagePositionPatient'].nunique() != len(df) and not (df['venc'] != 0).any()]

        single_mra = [df for df in single_phase_list if df['rr'].max() == 0]
        single_whole_heart = [df for df in single_phase_list if df['rr'].max() > 0]
        multi_mra = [df for df in multi_phase_list if df['rr'].max() == 0]
        cine_4d = [df for df in multi_phase_list if (df['rr'].max() > 0) & (df['InstanceNumber'].nunique() == 0)]

        # Update the sort_dict_3D 
        sort_dict_3D = self.update_sort_dict(sort_dict_3D, 3, single_mra, 'Single MRA', flow_flag=0, cine_flag=0, stack_flag = 'NA')
        sort_dict_3D = self.update_sort_dict(sort_dict_3D, 3, single_whole_heart, 'Single Whole-Heart', flow_flag=0, cine_flag=0, stack_flag = 'NA')
        sort_dict_3D = self.update_sort_dict(sort_dict_3D, 3, multi_mra, 'Multi MRA', flow_flag=0, cine_flag=0, stack_flag = 'NA')
        sort_dict_3D = self.update_sort_dict(sort_dict_3D, 3, flow_4d, '4D Flow', flow_flag=1, cine_flag=1, stack_flag = 'NA')
        sort_dict_3D = self.update_sort_dict(sort_dict_3D, 3, cine_4d, '4D Cine', flow_flag=0, cine_flag=1, stack_flag = 'NA')
        return sort_dict_3D