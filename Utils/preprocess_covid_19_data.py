import os
import numpy as np
from Utils.data_io_3D import ScanWrapper
import pandas as pd
from Utils.misc import get_logger
from tqdm import tqdm
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


logger = get_logger(os.path.basename(__file__))


class PreprocessDatasetCOVID19:
    def __init__(self):
        # self.data_root = '/nfs/masi/xuk9/Projects/COVID19_severity_score/Data'
        self.data_root = '/home/local/VANDERBILT/dongc1/Desktop/Projects/Knee'
        self.label_csv = os.path.join(self.data_root, 'labels.csv')

    def add_lesion_score(self):
        logger.info(f'Load {self.label_csv}')
        label_df = pd.read_csv(self.label_csv)

        ct_dir = os.path.join(self.data_root, 'ct')
        lung_mask_dir = os.path.join(self.data_root, 'lung_mask')
        lesion_seg_dir = os.path.join(self.data_root, 'lesion_seg')

        for index, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
            ct_filename = row['ct_file_name']
            lesion_filename = ct_filename.replace('.nii.gz', '_seg.nii.gz')
            ct_path = os.path.join(ct_dir, ct_filename)
            lung_mask_path = os.path.join(lung_mask_dir, ct_filename)
            lesion_seg_path = os.path.join(lesion_seg_dir, lesion_filename)

            if os.path.exists(lesion_seg_path):
                ct_image = ScanWrapper(ct_path).get_data()
                lung_mask = ScanWrapper(lung_mask_path).get_data()
                lesion_seg = ScanWrapper(lesion_seg_path).get_data()

                assert lung_mask.shape == lesion_seg.shape

                lesion_mask = np.where((lesion_seg == 1) & (lung_mask > 0), 1, 0)
                ho_mask = np.where((ct_image > -200) & (lesion_seg == 1) & (lung_mask > 0), 1, 0)

                po_score = 100 * np.count_nonzero(lesion_mask) / np.count_nonzero(lung_mask)
                pho_score = 100 * np.count_nonzero(ho_mask) / np.count_nonzero(lung_mask)

                label_df.loc[index, 'Annotation'] = True  # Have annotation
                label_df.loc[index, 'PO'] = po_score
                label_df.loc[index, 'PHO'] = pho_score
            else:
                label_df.loc[index, 'Annotation'] = False

        logger.info(f'Update {self.label_csv}')
        label_df.to_csv(self.label_csv, index=False)

    def get_demographic_information(self):
        """
        Get cohort demographic summary:
        1. Sex
        2. Age (< 18, 18 - 55, 55 - 80, > 80)
        3. Chest X-Ray view position: pa, ap, unknown
        4. PCR results: At least one positive / all negative
        :return:
        """
        logger.info(f'Load {self.label_csv}')
        label_df = pd.read_csv(self.label_csv, dtype={'Annotation': np.bool})
        annotate_df = label_df.loc[label_df['Annotation']]
        logger.info(f'Number of case with segmentation: {len(annotate_df.index)}')

        patient_list = annotate_df['patient_id'].to_list()
        patient_list = list(set(patient_list))
        logger.info(f'Number of unique subjects: {len(patient_list)}')

        # Remove duplicated records
        logger.info(f'Remove duplicates')
        annotate_df = annotate_df.drop_duplicates(subset=["patient_id"])
        logger.info(f'After removal: {len(annotate_df.index)}')

        gender_list = annotate_df['gender'].to_list()
        gender_counter = Counter(gender_list)
        logger.info('Gender:')
        for gender in gender_counter:
            logger.info(f'  {gender} - {gender_counter[gender]}')

        age_list = annotate_df['age'].to_list()
        age_bins = [16, 55, 80]
        age_inds = np.digitize(np.array(age_list), age_bins)
        age_counter = Counter(age_inds)
        age_dict = {
            0: "[0, 16)",
            1: "[18, 55)",
            2: "[55, 80)",
            3: "[80, -)"
        }
        logger.info('Age:')
        for age_ind in range(len(age_dict)):
            logger.info(f'  {age_dict[age_ind]} - {age_counter[age_ind]}')

        pcr_result_list = annotate_df['covid'].to_list()
        pcr_result_counter = Counter(pcr_result_list)
        logger.info(f'PCR result')
        for pcr_result in pcr_result_counter:
            logger.info(f'  {pcr_result} - {pcr_result_counter[pcr_result]}')

    def plot_severity_score_dist(self):
        logger.info(f'Load {self.label_csv}')
        label_df = pd.read_csv(self.label_csv, dtype={'Annotation': np.bool})
        annotate_df = label_df.loc[label_df['Annotation']]
        long_df = pd.melt(annotate_df, id_vars=['ID'], value_vars=['PO', 'PHO'])
        fig, ax = plt.subplots(figsize=(7, 6))
        # sns.displot(data=long_df, x='value', hue='variable', kind='kde', ax=ax)
        sns.histplot(data=long_df, x='value', hue='variable', multiple="dodge", shrink=0.8, ax=ax, kde=True)
        ax.set_xlabel('Severity Score (%)')
        legend = ax.get_legend()
        handles = legend.legendHandles
        legend.remove()
        ax.legend(handles, ['PO', 'PHO'], title='Score')
        out_png = os.path.join(self.data_root, 'severity_dist.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

    def plot_correlation_two_plot(self):
        logger.info(f'Load {self.label_csv}')
        label_df = pd.read_csv(self.label_csv, dtype={'Annotation': np.bool})
        annotate_df = label_df.loc[label_df['Annotation']]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(annotate_df['PO'].to_list(), annotate_df['PHO'].to_list(), color='r', alpha=0.4)
        ax.plot([0, 100], [0, 100], c='b', linewidth=2, linestyle='dashed')
        ax.set_xlabel('PO Score (%)')
        ax.set_ylabel('PHO Score (%)')

        out_png = os.path.join(self.data_root, 'severity_correlation.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

    def generate_sample_cases(self):
        """
        Get the sample's combined masks. Lung mask + lesion mask + high opacity mask
        :return:
        """
        logger.info(f'Load {self.label_csv}')
        label_df = pd.read_csv(self.label_csv, dtype={'Annotation': np.bool})
        annotate_df = label_df.loc[label_df['Annotation']]

        out_combined_mask_dir = os.path.join(self.data_root, 'combined_mask')
        os.makedirs(out_combined_mask_dir, exist_ok=True)
        out_combined_sample_ct_dir = os.path.join(self.data_root, 'combined_sample_ct')
        os.makedirs(out_combined_sample_ct_dir, exist_ok=True)

        lung_mask_dir = os.path.join(self.data_root, 'lung_mask')
        lesion_seg_dir = os.path.join(self.data_root, 'lesion_seg')
        ct_dir = os.path.join(self.data_root, 'ct')

        patient_list = ['P0337', 'P0307', 'P0431', 'P0013']
        for patient_id in tqdm(patient_list, total=len(patient_list)):
            # for patient_id in patient_list:
            patient_df = annotate_df.loc[annotate_df['ID'] == patient_id]
            ct_filename = patient_df.iloc[0]['ct_file_name']
            # seg_filename = ct_filename.replace('.nii.gz', '_seg.nii.gz')
            # ct_object = ScanWrapper(os.path.join(ct_dir, ct_filename))
            # ct_image = ct_object.get_data()
            # lung_mask = ScanWrapper(os.path.join(lung_mask_dir, ct_filename)).get_data()
            # lesion_seg_mask = ScanWrapper(os.path.join(lesion_seg_dir, seg_filename)).get_data()
            #
            # combined_mask = np.zeros(lesion_seg_mask.shape, dtype=int)
            # combined_mask[lung_mask > 0] = 1
            # combined_mask[(lung_mask > 0) & (lesion_seg_mask == 1)] = 2
            # combined_mask[(lung_mask > 0) & (lesion_seg_mask == 1) & (ct_image > -200)] = 3
            #
            # out_path = os.path.join(out_combined_mask_dir, f'{patient_id}.nii.gz')
            # ct_object.save_scan_same_space(out_path, combined_mask)

            ct_ln_path = os.path.join(out_combined_sample_ct_dir, f'{patient_id}.nii.gz')
            ln_cmd = f'ln -sf {os.path.join(ct_dir, ct_filename)} {ct_ln_path}'
            os.system(ln_cmd)

    def generate_qa_flag(self):
        """
        Need to filter out the pediatric cases (age < 16).
        :return:
        """
        logger.info(f'Load {self.label_csv}')
        label_df = pd.read_csv(self.label_csv, dtype={'Annotation': np.bool})

        age_thres = 16
        label_df['qa_result'] = np.where(label_df['age'] >= age_thres, True, False)
        qa_pass_df = label_df.loc[label_df['qa_result']]
        logger.info(f'Number of failed QA: {len(label_df.index) - len(qa_pass_df.index)}')
        logger.info(f'Update {self.label_csv}')
        label_df.to_csv(self.label_csv)

    def train_valid_split(self):
        """
        :return:
        """
        logger.info(f'Load {self.label_csv}')
        label_df = pd.read_csv(self.label_csv)

        # Train / valid / test split.
        # num_total = len(label_df.loc[label_df['Annotation'] & label_df['qa_result']].index)
        num_total = len(label_df.index)
        n_train = int(round(0.7 * num_total))
        n_valid = int(round(0.1 * num_total))
        train_df = label_df.sample(n=n_train, random_state=63)
        label_df['split_train'] = np.where(label_df.index.isin(train_df.index), 1, 0)
        valid_df = label_df.loc[(label_df['split_train'] == 0)].sample(n=n_valid, random_state=63)
        label_df['split_valid'] = np.where(label_df.index.isin(valid_df.index), 1, 0)
        test_df = label_df.loc[(label_df['split_train'] == 0) &
                               (label_df['split_valid'] == 0)]
        label_df['split_test'] = np.where(label_df.index.isin(test_df.index), 1, 0)

        # Get statistics
        logger.info(f'train: {len(train_df)}')
        logger.info(f'valid: {len(valid_df)}')
        logger.info(f'test: {len(test_df)}')

        logger.info(f'Update {self.label_csv}')
        label_df.to_csv(self.label_csv, index=False)


if __name__ == '__main__':
    data_processor = PreprocessDatasetCOVID19()
    # data_processor.add_lesion_score()
    # data_processor.get_demographic_information()
    # data_processor.plot_severity_score_dist()
    # data_processor.plot_correlation_two_plot()
    # data_processor.generate_sample_cases()
    # data_processor.generate_qa_flag()
    data_processor.train_valid_split()