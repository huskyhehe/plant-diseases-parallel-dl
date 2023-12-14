base = "/home/zhou.he1/final/"
dataset_path = "new-plant-diseases-dataset"
train_dir = dataset_path + "/train"
valid_dir = dataset_path + "/valid"
test_dir = dataset_path + "/test"

# common mean and std for image datasets
# mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# plant dataset mean and std
mean, std = [0.4756, 0.5001, 0.4263], [0.2166, 0.1957, 0.2323]

num_classes = 38
input_shape = (3, 256, 256)
batch_size = 64
