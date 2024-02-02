from data_preparation import Dataset

dataset = Dataset()

print(dataset.train_data.__len__())
print(dataset.train_subset.__len__())

print(dataset.test_data.__len__())
print(dataset.test_subset.__len__())