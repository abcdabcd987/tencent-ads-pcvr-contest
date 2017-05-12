from __future__ import print_function
import os
from tqdm import tqdm

from Dataset import Dataset, get_num_lines


class TencentAdsPcvrContestPreData(Dataset):
    block_size = 200000
    train_num_of_parts = 18
    test_num_of_parts = 2
    valid_num_of_parts = 2
    train_size = 0
    test_size = 0
    train_pos_samples = 0
    test_pos_samples = 0
    train_neg_samples = 0
    test_neg_samples = 0
    train_pos_ratio = 0
    test_pos_ratio = 0
    initialized = 0
    max_length = 24
    num_features = 75387
    feat_names = ['_', 'adID', 'advertiserID', 'age', 'appCategory', 'appID', 'appPlatform', 'camgaignID', 
        'clickHour', 'clickWeekday', 'connectionType', 'creativeID', 'education', 'gender', 'haveBaby', 'hometown', 
        'installedApp', 'marriageStatus', 'positionID', 'positionType', 'residence', 
        'sitesetID', 'telecomsOperator', 'userID']
    feat_min = [0, 1, 2742, 2831, 2913, 2928, 2979, 2982, 3583, 3608, 3616, 3622, 7886, 7895, 7899, 7907, 
        8273, 61901, 61906, 64838, 64845, 65239, 65243, 65248]
    feat_max = [0, 2741, 2830, 2912, 2927, 2978, 2981, 3582, 3607, 3615, 3621, 7885, 7894, 7898, 7906, 8272, 
        61900, 61905, 64837, 64844, 65238, 65242, 65247, 75385]
    feat_sizes = [feat_max[i] - feat_min[i] + 1 for i in range(max_length)]
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tencent-ads-pcvr-contest-pre')
    raw_data_dir = os.path.join(data_dir, 'raw')
    feature_data_dir = os.path.join(data_dir, 'feature')
    hdf_data_dir = os.path.join(data_dir, 'hdf')

    def __init__(self, initialized=True, dir_path='../tencent-ads-pcvr-contest-pre', max_length=None, num_features=None,
                 block_size=200000):
        """
        collect meta information, and produce hdf files if not exists
        :param initialized: write feature and hdf files if True
        :param dir_path: 
        :param max_length: 
        :param num_features: 
        :param block_size: 
        """
        self.initialized = initialized
        if not self.initialized:
            print('Got raw ads_pcvr_contest_pre data, initializing...')
            self.raw_data_dir = os.path.join(dir_path, 'raw')
            self.feature_data_dir = os.path.join(dir_path, 'feature')
            self.hdf_data_dir = os.path.join(dir_path, 'hdf')
            self.max_length = max_length
            self.num_features = num_features
            self.block_size = block_size
            if self.max_length is None or self.num_features is None:
                print('Getting the maximum length and # features...')
                min_combine_length, max_combine_length, max_combine_feature = self.get_length_and_feature_number(
                    os.path.join(self.raw_data_dir, 'combine.txt'))
                min_test_length, max_test_length, max_test_feature = self.get_length_and_feature_number(
                    os.path.join(self.raw_data_dir, 'test.txt'))
                self.max_length = max(max_combine_length, max_test_length)
                self.num_features = max(max_combine_feature, max_test_feature) + 1
            print('max length = %d, # features = %d' % (self.max_length, self.num_features))

            self.combine_num_of_parts = self.raw_to_feature(raw_file='combine.txt',
                                                          input_feat_file='combine_input.txt',
                                                          output_feat_file='combine_output.txt')
            self.feature_to_hdf(num_of_parts=self.combine_num_of_parts,
                                file_prefix='combine',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir,
                                input_columns=self.feat_names,
                                output_columns=['convert'])

            self.train_num_of_parts = self.raw_to_feature(raw_file='train.txt',
                                                          input_feat_file='train_input.txt',
                                                          output_feat_file='train_output.txt')
            self.feature_to_hdf(num_of_parts=self.train_num_of_parts,
                                file_prefix='train',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir,
                                input_columns=self.feat_names,
                                output_columns=['convert'])
            ''' 
            self.valid_num_of_parts = self.raw_to_feature(raw_file='valid.txt',
                                                          input_feat_file='valid_input.txt',
                                                          output_feat_file='valid_output.txt')
            self.feature_to_hdf(num_of_parts=self.valid_num_of_parts,
                                file_prefix='valid',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir,
                                input_columns=self.feat_names,
                                output_columns=['convert'])
            '''
            self.test_num_of_parts = self.raw_to_feature(raw_file='test.txt',
                                                         input_feat_file='test_input.txt',
                                                         output_feat_file='test_output.txt')
            self.feature_to_hdf(num_of_parts=self.test_num_of_parts,
                                file_prefix='test',
                                feature_data_dir=self.feature_data_dir,
                                hdf_data_dir=self.hdf_data_dir,
                                input_columns=self.feat_names,
                                output_columns=['convert'])
        
        print('Got hdf tencent-ads-pcvr-contest-pre data set, getting metadata...')
        self.train_size, self.train_pos_samples, self.train_neg_samples, self.train_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'train', self.train_num_of_parts)
        self.test_size, self.test_pos_samples, self.test_neg_samples, self.test_pos_ratio = \
            self.bin_count(self.hdf_data_dir, 'test', self.test_num_of_parts)
        #self.valid_size, self.valid_pos_samples, self.valid_neg_samples, self.valid_pos_ratio = \
            #self.bin_count(self.hdf_data_dir, 'valid', self.valid_num_of_parts)
        print('Initialization finished!')

    def raw_to_feature(self, raw_file, input_feat_file, output_feat_file):
        """
        Transfer the raw data to feature data. using static method is for consistence 
            with multi-processing version, which can not be packed into a class
        :param raw_file: The name of the raw data file.
        :param input_feat_file: The name of the feature input data file.
        :param output_feat_file: The name of the feature output data file.
        :return:
        """
        print('Transferring raw', raw_file, 'data into feature', raw_file, 'data...')
        raw_file = os.path.join(self.raw_data_dir, raw_file)
        feature_input_file_name = os.path.join(self.feature_data_dir, input_feat_file)
        feature_output_file_name = os.path.join(self.feature_data_dir, output_feat_file)
        line_no = 0
        cur_part = 0
        fin = open(feature_input_file_name + '.part_' + str(cur_part), 'w')
        fout = open(feature_output_file_name + '.part_' + str(cur_part), 'w')
        with open(raw_file) as rin:
            for line in tqdm(rin, total=get_num_lines(raw_file)):
                line_no += 1
                if self.block_size is not None and line_no % self.block_size == 0:
                    fin.close()
                    fout.close()
                    cur_part += 1
                    fin = open(feature_input_file_name + '.part_' + str(cur_part), 'w')
                    fout = open(feature_output_file_name + '.part_' + str(cur_part), 'w')

                fields = line.strip().split()
                y_i = fields[0]
                X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
                fout.write(y_i + '\n')
                first = True

                if len(X_i) > self.max_length:
                    X_i = X_i[:self.max_length]
                elif len(X_i) < self.max_length:
                    X_i.extend([self.num_features - 1] * (self.max_length - len(X_i)))

                for item in X_i:
                    if first:
                        fin.write(str(item))
                        first = False
                    else:
                        fin.write(',' + str(item))
                fin.write('\n')
        fin.close()
        fout.close()
        return cur_part + 1

    @staticmethod
    def get_length_and_feature_number(file_name):
        """
        Get the min_length max_length and max_feature of data.
        :param file_name: The file name of input data.
        :return: the tuple (min_length, max_length, max_feature)
        """
        max_length = 0
        min_length = 99999
        max_feature = 0
        line_no = 0
        with open(file_name) as fin:
            for line in tqdm(fin, total=get_num_lines(file_name)):
                line_no += 1
                if line_no % 100000 == 0:
                    print('%d lines finished.' % (line_no))
                fields = line.strip().split()
                X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
                max_feature = max(max_feature, max(X_i))
                max_length = max(max_length, len(X_i))
                min_length = min(min_length, len(X_i))
        return min_length, max_length, max_feature

# if __name__ == '__main__':
#     TencentAdsPcvrContestPreData(initialized=False, max_length=863, num_features=75387)
