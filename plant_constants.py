dataset_path = "new-plant-diseases-dataset"
train_dir = dataset_path + "/train"
valid_dir = dataset_path + "/valid"
test_dir = dataset_path + "/test"

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
num_classes = 38
input_shape = (3, 256, 256)
batch_size = 64