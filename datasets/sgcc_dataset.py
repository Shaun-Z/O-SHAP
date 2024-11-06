import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

from datasets.base_dataset import BaseDataset

class SgccDataset(BaseDataset):
    """A dataset class for SGCC Theft dataset.

    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--normalize', action='store_true', help='if specified, normalize the data')
        parser.add_argument('--datafile_name', type=str, default='data.csv', help='name of the data file')
        parser.add_argument('--maskfile_name', type=str, default='mask.csv', help='name of the mask file')
        parser.add_argument('--val_percent', type=float, default=0.1, help='percentage of the data to use as validation set')
        return parser
    
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)  # call the default constructor of BaseDataset
        self.phase = opt.phase # get the phase: train, val, test
        self.dataroot = os.path.join(opt.dataroot)
        self.data = os.path.join(self.dataroot, opt.datafile_name)
        self.mask = os.path.join(self.dataroot, opt.maskfile_name)
        self.normalize = opt.normalize
        self.val_percent = opt.val_percent

        prepare_data_rolling_window()

        self.load_data()

    def load_data(self):
        """Load the data from the csv file"""
        df_data = pd.read_csv(self.data)
        df_mask = pd.read_csv(self.mask)

        data_train, data_test, mask_train, mask_test = train_test_split(df_data, df_mask, test_size=self.val_percent, random_state=0)

        if self.phase == 'train':
            df_data = data_train
            df_mask = mask_train
        elif self.phase == 'val':
            df_data = data_test
            df_mask = mask_test
        else:
            raise NotImplementedError(f'Phase {self.phase} is not implemented')

        self.max = df_data.max()
        self.min = df_mask.min()

        df_data_normalized = (df_data - self.min)/(self.max - self.min)

        self.data_tensor = torch.tensor(df_data_normalized.values).unsqueeze(1).float() if self.normalize else torch.tensor(df_data.values).unsqueeze(1).float()
        self.mask_tensor = torch.tensor(df_mask.values)

    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, idx):

        data = self.data_tensor[idx]
        mask = self.mask_tensor[idx]

        assert data.size(-1) == mask.size(-1), \
            f'Data\'s and mask\'s feature_dim should be the same size, but are {data.size} and {mask.size}'

        return {
            'data': data.contiguous(),
            'mask': mask.contiguous()
        }

# %%
def type1_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
    alpha = np.random.uniform(*alpha_range)
    end_point = start_point + duration
    data[:, :, start_point:end_point] *= alpha
    return data

def type2_attack(data, start_point, duration):
    end_point = start_point + duration
    sigma_range = (0, np.max(data))
    sigma = np.random.uniform(*sigma_range)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, start_point:end_point] = np.clip(data[i, j, start_point:end_point], None, sigma)
    return data

def type3_attack(data, start_point, duration, gamma):
    end_point = start_point + duration
    data[:, :, start_point:end_point] *= (1-gamma)
    data = np.maximum(data, 0)  # Ensure no negative values
    return data

def type4_attack(data, start_point, duration):
    end_point = start_point + duration
    data[:, :, start_point:end_point] = 0
    return data

def type5_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
    end_point = start_point + duration
    for t in range(start_point, end_point):
        alpha_t = np.random.uniform(*alpha_range)
        data[:, :, t] *= alpha_t
    return data

def type6_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
    end_point = start_point + duration
    for t in range(start_point, end_point):
        alpha_t = np.random.uniform(*alpha_range)
        daily_average = np.mean(data[:, :, :], axis=2, keepdims=True)
        data[:, :, t] = alpha_t * daily_average[:, :, 0]
    return data

'''
定义两个函数：
has_continuous_zeros(row, threshold=30)：检查一行数据中是否有连续超过 threshold 个小于等于 0.17 的值。如果有，返回 True，否则返回 False。
'''
def has_continuous_zeros(row, threshold=30):
    consecutive_zeros = 0
    for value in row:
        if value <= 0.17:
            consecutive_zeros += 1
            if consecutive_zeros > threshold:
                return True
        else:
            consecutive_zeros = 0
    return False
'''
has_continuous_nans(row, threshold=2)：检查一行数据中是否有连续超过 threshold 个 NaN 值。如果有，返回 True，否则返回 False。
'''
def has_continuous_nans(row, threshold=3):
    consecutive_nans = 0
    for value in row:
        if pd.isna(value):
            consecutive_nans += 1
            if consecutive_nans > threshold:
                return True
        else:
            consecutive_nans = 0
    return False

def prepare_data_rolling_window(data_path: str = "./data/sgcc/data.csv"):
    original_data = pd.read_csv(data_path)
    original_data_theft = original_data[original_data.FLAG == 1]
    original_data_normal = original_data[original_data.FLAG == 0]

    theft_condition_nan_or_zero = original_data_theft.iloc[:, 2:].isnull() | (original_data_theft.iloc[:, 2:] == 0)
    original_data_theft_nan_or_zero = original_data_theft[theft_condition_nan_or_zero.all(axis=1)]

    normal_condition_nan_or_zero = original_data_normal.iloc[:, 2:].isnull() | (original_data_normal.iloc[:, 2:] == 0)
    original_data_normal_nan_or_zero = original_data_normal[normal_condition_nan_or_zero.all(axis=1)]

    usable_theft = original_data_theft[~theft_condition_nan_or_zero.all(axis=1)]
    usable_normal = original_data_normal[~normal_condition_nan_or_zero.all(axis=1)]

    usable_theft = usable_theft.drop(columns=['CONS_NO', 'FLAG'])
    usable_normal = usable_normal.drop(columns=['CONS_NO', 'FLAG'])


    usable_theft = usable_theft.T
    usable_normal = usable_normal.T

    usable_theft.index = pd.to_datetime(usable_theft.index, format='%Y/%m/%d')
    usable_normal.index = pd.to_datetime(usable_normal.index, format='%Y/%m/%d')

    usable_theft = usable_theft.sort_index()
    usable_normal = usable_normal.sort_index()


    usable_theft_2016 = usable_theft[usable_theft.index.year == 2016]
    usable_normal_2016 = usable_normal[usable_normal.index.year == 2016]
    usable_theft_2016 = usable_theft_2016.T
    usable_normal_2016 = usable_normal_2016.T
    print(usable_normal_2016.shape)

    usable_theft = usable_theft.T
    usable_normal = usable_normal.T
    ###################### preprocessing for 2016 normal data

    usable_normal_2016 = usable_normal_2016[~usable_normal_2016.apply(has_continuous_zeros, axis=1)]
    print(usable_normal_2016.shape)

    usable_normal_2016 = usable_normal_2016[~usable_normal_2016.apply(has_continuous_nans, axis=1)] #### can relax to larger continous value
    print(usable_normal_2016.shape)



    # Transpose the DataFrame so that days become rows (needed for rolling)
    df_transposed = usable_normal_2016.T
    # Apply the rolling window (e.g., 7-day) and calculate the rolling mean
    rolling_mean = df_transposed.rolling(window=7, min_periods=1).mean().T

    # Calculate the rolling standard deviation
    rolling_std = df_transposed.rolling(window=7, min_periods=1).std().T
    
    outliers = (usable_normal_2016 - rolling_mean).abs() > (2.2 * rolling_std)
    # Replace outliers with -99
    df_with_replaced_outliers = usable_normal_2016.mask(outliers, -99)
    # Remove rows with any -99 values
    df_cleaned = df_with_replaced_outliers[~df_with_replaced_outliers.eq(-99).any(axis=1)]


    df_cleaned_interpolated = df_cleaned.interpolate(method='linear', axis=1)
    df_cleaned_interpolated = df_cleaned_interpolated[~df_cleaned_interpolated.isnull().any(axis=1)]

    df_cleaned_max_min_diff = df_cleaned_interpolated.max(axis=1) - df_cleaned_interpolated.min(axis=1)


    # Shuffle the DataFrame rows
    df_shuffled = df_cleaned_interpolated.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the DataFrame into two equal parts
    df1 = df_shuffled.iloc[:6000]
    df2 = df_shuffled.iloc[6000:]



    ####################################################################
    # 为每个用户生成两个0到1之间的随机数
    start_rates = np.random.rand(len(df1))
    end_rates = np.random.rand(len(df1))
    # 确保 end_rates 对应的位置上的数字比 start_rates 上的大
    mask = end_rates < start_rates
    start_rates[mask], end_rates[mask] = end_rates[mask], start_rates[mask]

    # 创建一个新的DataFrame来存储攻击的结果
    attack_df = df1.copy()
    # 创建一个新的DataFrame来存储标签
    label_df = pd.DataFrame(0, index=df1.index, columns=df1.columns)

    print(f"Time length: {len(df1.columns)}")

    '''
    def type1_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
    def type2_attack(data, start_point, duration, sigma):
    def type3_attack(data, start_point, duration, gamma):
    def type4_attack(data, start_point, duration):
    def type5_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
    def type6_attack(data, start_point, duration, alpha_range=(0.2, 0.8)):
    '''

    # 对每个用户施加攻击
    err=[]
    for i in range(len(df1)):
        # 计算开始索引和结束索引
        start_index = int(start_rates[i] * len(df1.columns))
        end_index = int(end_rates[i] * len(df1.columns))

        # 计算持续时间
        duration = end_index - start_index

        # 将原始的DataFrame转换为数组
        data_array = np.expand_dims(df1.iloc[i].values, axis=0)
        data_array_copy = df1.copy().iloc[i].values

        # 在数组上施加攻击
        # attack_result = type1_attack(np.expand_dims(data_array, axis=1), start_index, duration)

        # alpha, attack_result = type1_attack(np.expand_dims(data_array, axis=1), start_index, duration)
        # print(f"{alpha}, {start_index} - {end_index} : {duration}")

        # attack_result = type2_attack(np.expand_dims(data_array, axis=1), start_index, duration, 0.1)
        attack_result = type3_attack(np.expand_dims(data_array, axis=1), start_index, duration, 0.1)

        # print( np.sum(data_array_copy- attack_result[0,0,:]))
        err.append(np.sum(data_array_copy- attack_result[0,0,:]))

        if np.sum(data_array_copy- attack_result[0,0,:]) > 100:###一年窃电100kwh以上
            # 将攻击结果存储到attack_df中
            attack_df.iloc[i, :] = attack_result.squeeze()

            # 更新标签
            label_df.iloc[i, start_index:end_index] = 1

    zy = label_df[(label_df != 0).any(axis=1)] 
    zx = attack_df[attack_df.index.isin(zy.index)] 

    # 将攻击的结果和标签存储为两个CSV文件
    # zx.to_csv('zx3.csv', index=False)
    zy.to_csv('./data/sgcc/label_attack3.csv', index=False)

    from sklearn.preprocessing import MinMaxScaler
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    # Apply MinMaxScaler to each row
    df_normalized = zx.copy()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized.T).T, columns=df_normalized.columns)

    df_normalized.to_csv('./data/sgcc/data_attack3_normalized.csv', index=False)

    df2_normalized = df2.copy()
    df2_normalized = pd.DataFrame(scaler.fit_transform(df2_normalized.T).T, columns=df2_normalized.columns)
    df2_normalized.to_csv('./data/sgcc/data_normal3_normalized.csv', index=False)

    label_df2 = pd.DataFrame(0, index=df2.index, columns=df2.columns)
    label_df2.to_csv('./data/sgcc/label_normal3.csv', index=False)

    label_df4 = pd.DataFrame(0, index=df2.index, columns=df2.columns)
    label_df4.iloc[:, -12:] = [0,1,1,1,0,1,1,1,0,1,1,1]
    label_df4.to_csv('./data/sgcc/label_pseudo_normal3.csv', index=False)


    zx3_normalized = pd.read_csv(f'./data/sgcc/data_attack3_normalized.csv')
    normal3_normalized = pd.read_csv(f'./data/sgcc/data_normal3_normalized.csv')
    # Rename the columns of df2 to match df1
    normal3_normalized.columns = zx3_normalized.columns

    zy3 = pd.read_csv(f'./data/sgcc/label_attack3.csv') 
    normal3_normalized_label = pd.read_csv(f'./data/sgcc/label_normal3.csv')
    normal3_normalized_label.columns = zy3.columns

    combined_dfx = pd.concat([zx3_normalized, normal3_normalized], ignore_index=True)#
    combined_dfy = pd.concat([zy3, normal3_normalized_label], ignore_index=True)#
    combined_dfx.to_csv('./data/sgcc/combined_dfx.csv', index=False)
    combined_dfy.to_csv('./data/sgcc/combined_dfy.csv', index=False)

    normal3_normalized_sudolabel = pd.read_csv(f'./data/sgcc/label_pseudo_normal3.csv')
    normal3_normalized_sudolabel.columns = zy3.columns
    combined_dfy_sudo = pd.concat([zy3, normal3_normalized_sudolabel], ignore_index=True)#
    combined_dfy_sudo.to_csv('./data/sgcc/combined_dfy_pseudo.csv', index=False)


