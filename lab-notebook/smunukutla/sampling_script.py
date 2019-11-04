import os
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from spectra_ml.io_ import load_spectra_metadata

spectrum_len = 500 # automate this
data_dir = os.environ['DATA_DIR']
stddata_path = os.path.join(data_dir, "StdData-" + str(spectrum_len))
metadata = load_spectra_metadata(os.path.join(stddata_path,"spectra-metadata.csv"))

metadata = metadata[metadata['value_type'] == "reflectance"]
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]
metadata = metadata[metadata['raw_data_path'].str.contains("ChapterM")] # add in ChapterS Soils and Mixtures later

nan_records = set()
x = 0
for i in range(metadata.shape[0]):
    spectrum = pd.read_csv(os.path.join(stddata_path,"{}.csv".format(str(metadata.iloc[i, 0]))))
    for j in range(spectrum.shape[0]):
        if np.isnan(spectrum.iloc[j, 1]):
            # print(str(metadata.iloc[i, 3]) + " " + str(metadata.iloc[i, 0]) + " " + str(metadata.iloc[i, 2]))
            nan_records.add(metadata.iloc[i, 0])
            x += 1
            break

def removenans(metadata, nan_set):
    indices = []
    ret = []
    for row in range(metadata.iloc[:, 0].shape[0]):
        indices.append(metadata.index[row])
        if metadata.iat[row, 0] in nan_set:
            ret.append(False)
        else:
            ret.append(True)
    ret = pd.Series(ret, index=indices)
    return ret

metadata = metadata[removenans(metadata, nan_records)]

# using the minerals with the most amount of data
metadata.sort_values('material',inplace=True)

frame = pd.DataFrame(columns=['material', 'count'])

series = metadata['material']
series = series.apply(lambda x: x.split(" ")[0])
series = series.value_counts()

frame['count'] = series.values
frame['material'] = series.index

# frame = frame[frame['count'] >= 12]

frame = frame[:10]

# dictionary = {"Actinolite": 0, "Alunite": 1, "Chlorite": 2, "Topaz": 3, "Olivine": 4}

dictionary = {frame.iloc[:, 0].tolist()[i] : i for i in range(len(frame.iloc[:, 0].tolist()))}
num_minerals = len(dictionary)

record_nums_old = []
y_old = []
spectrum_names_old = []

for i in range(metadata.shape[0]): # add dictionary/clean up metadata
    data = metadata.iloc[i, :]
    mineral_name = data['material'].split(" ")[0]
    if mineral_name in dictionary.keys():
        record_nums_old.append(data[0])
        spectrum_names_old.append(mineral_name)
        y_old.append(dictionary[mineral_name])

record_nums = record_nums_old.copy()
record_nums.sort()

y = []
spectrum_names = []

for i in record_nums:
    ind = record_nums_old.index(i)
    spectrum_names.append(spectrum_names_old[ind])
    y.append(y_old[ind])

with open("data.csv", "w") as fi:
    writer = csv.writer(fi)
    writer.writerow(list(range(0, len(y))))
    writer.writerow(record_nums)
    writer.writerow(spectrum_names)
    writer.writerow(y)
    # y = np.reshape(y, (len(y), 1))

fi.close()

# find the train test split
num_samples = len(y)

num_runs = 20
print(num_runs)
print(num_minerals)

dev_and_test_y = pd.DataFrame(data=np.transpose(y))

for i in range(num_runs):
    sample_indices = list(range(0, num_samples))

    train_set_indices, dev_and_test = train_test_split(sample_indices, test_size=0.4, stratify=y)
    dev_set_indices, test_set_indices = train_test_split(dev_and_test, test_size=0.5, stratify=dev_and_test_y.iloc[dev_and_test])
    # train_and_dev, test_set_indices = train_test_split(sample_indices, test_size=0.2, stratify=y)
    # train_set_indices, dev_set_indices = train_test_split(train_and_dev, test_size=0.25, stratify=y[train_and_dev])
    print(train_set_indices)
    print(test_set_indices)
    print(dev_set_indices)
