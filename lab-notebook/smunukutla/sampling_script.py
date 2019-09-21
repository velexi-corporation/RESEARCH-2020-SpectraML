import os
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

data_dir = os.environ['DATA_DIR']
stddata_path = os.path.join(data_dir,"Srikar-Standardized")
metadata = pd.read_csv(os.path.join(stddata_path,"spectra-metadata.csv"), sep="|", dtype={"spectrum_id":str})

metadata = metadata[metadata['value_type'] == "reflectance"]
metadata = metadata[~metadata['spectrometer_purity_code'].str.contains("NIC4")]
metadata = metadata[metadata['raw_data_path'].str.contains("ChapterM")] # add in ChapterS Soils and Mixtures later

record_nums = []
y = []
spectrum_names = []

act = 0
aln = 0
chl = 0

for i in range(metadata.shape[0]): # add dictionary/clean up metadata
    data = metadata.iloc[i, :]
    if data[2].find("Actinolite") != -1: # if material name contains actinolite
        record_nums.append(data[0])
        y.append(int(0))
        spectrum_names.append("Actinolite")
        act += 1
    elif data[2].find("Alun") != -1:
        record_nums.append(data[0])
        y.append(int(1))
        spectrum_names.append("Alunite")
        aln += 1
    elif (data[2].find("Chlorit") != -1 or data[2].find("Chlor.") != -1 or data[2].find("Chlor+") != -1 or data[2].find("Chl.") != -1):
        record_nums.append(data[0])
        y.append(int(2))
        spectrum_names.append("Chlorite")
        chl += 1

records_new = record_nums.copy()
records_new.sort()

y_new = []
names_new = []
for i in records_new:
    ind = record_nums.index(i)
    y_new.append(y[ind])
    names_new.append(spectrum_names[ind])

record_nums = records_new.copy()
y = y_new.copy()
spectrum_names = names_new.copy()

with open("data.csv", "w") as fi:
    writer = csv.writer(fi)
    writer.writerow(list(range(0, len(y))))
    writer.writerow(record_nums)
    writer.writerow(spectrum_names)
    writer.writerow(y)

fi.close()
# print(record_nums)
# print(spectrum_names)
# print(y)
# y = np.reshape(y, (len(y), 1))
num_samples = len(record_nums)
