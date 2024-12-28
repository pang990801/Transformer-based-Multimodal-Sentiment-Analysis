import os

import numpy as np
from mmsdk import mmdatasdk as md
import h5py

MOSI_PATH = "F:\MOSEI\model\data\MOSEI\cmumosi"
MOSI_DATASET = md.cmu_mosi

# cmumosi_highlevel=md.mmdataset(MOSI_DATASET.highlevel, MOSI_PATH)
# cmumosi_raw=md.mmdataset(MOSI_DATASET.raw, MOSI_PATH)
# cmumosi_labels=md.mmdataset(MOSI_DATASET.labels, MOSI_PATH)

visual_field = 'CMU_MOSI_Visual_Facet_42'
acoustic_field = 'CMU_MOSI_COVAREP'
text_field = 'CMU_MOSI_TimestampedWords'

features = [
    text_field,
    visual_field,
    acoustic_field
]

recipe = {feat: os.path.join(MOSI_PATH, feat) + '.csd' for feat in features}
dataset = md.mmdataset(recipe)


# we define a simple averaging function that does not depend on intervals
def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features


# first we align to words with averaging, collapse_function receives a list of functions
dataset.align(text_field, collapse_functions=[avg])

label_field = 'CMU_MOSI_Opinion_Labels'

# we add and align to lables to obtain labeled segments
# this time we don't apply collapse functions so that the temporal sequences are preserved
label_recipe = {label_field: os.path.join(MOSI_PATH, label_field + '.csd')}
dataset.add_computational_sequences(label_recipe, destination=None)
dataset.align(label_field)


# 保存 mmdataset 到 HDF5 格式
def save_dataset_to_hdf5(dataset, save_path):
    """
    保存 dataset 到 HDF5 格式
    """
    with h5py.File(save_path, "w") as hdf5_file:
        for field, comp_seq in dataset.computational_sequences.items():
            group = hdf5_file.create_group(field)  # 创建 HDF5 group 对应模态

            # 存储 features
            group.create_dataset("features", data=comp_seq.data['features'], compression="gzip")

            # 存储 intervals
            group.create_dataset("intervals", data=comp_seq.data['intervals'], compression="gzip")

            # 存储元数据（如果有）
            if comp_seq.metadata:
                metadata_group = group.create_group("metadata")
                for key, value in comp_seq.metadata.items():
                    # 将 metadata 转换为 numpy 数组或字符串保存
                    if isinstance(value, (list, np.ndarray)):
                        metadata_group.create_dataset(key, data=np.array(value), compression="gzip")
                    elif isinstance(value, str):
                        metadata_group.attrs[key] = value
                    else:
                        metadata_group.attrs[key] = str(value)

    print(f"Dataset successfully saved to {save_path}")


# 调用函数保存到 HDF5
save_path = "cmumosi/mosi.hdf5"
save_dataset_to_hdf5(dataset, save_path)
