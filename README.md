# Tensorflow-Utilities-Library
Most of Tensorflow's APIs are incredibly low level with horrendous documentation. This personal library will be used to solve a lot of common Tensorflow problems.


# Image
### Directory that handles working with image data.
## csv_to_dataset.py
**csv_to_dataset**
<br />description: Load in CSV data directly into a tensorflow dataset object. 
<br />CSV data must be in the format : <br />
2. First column is path to images. Needs to be path/image.jpg.
3. Second column are the labels. Labels are unchanged during the creation process.
<br />args:
* csv_path = posix path to CSV file
* batch_size = number : used to create dataset objects
* width = number : used to reshape image width
* height = number : used to reshape image height
* train_test_split = number : used to allocate data to training & validation
* aug_flag = boolean : used to include/ignore image augmentation during pre-processing
* labels_to_float = boolean : used if you'd like to convert your string labels to a list of floats

*functions*
<br />
* **dunder call overridden:** Can now call on the class itself to retrieve dataset & validation set as long as class has been properly initialized. Return type => <tensorflow.python.data.ops.dataset_ops.BatchDataset>

*examples*
<br />
initialization: loader_obj = DataLoader(CSV, BATCH_SIZE, width=224, height=224, train_test_split=0.9, aug_flag=True)
<br />
dataset retrieval: train_data, val_data = loader_obj()


## csv_utils.py
**scale_bounding_boxes**
<br />description: Scales your bounding box information from 0-255 to 0-1. 
<br />args:
* csv_path = posix path to CSV file
**denorm_numpy_image**
<br />description: Reverts your numpy image tensor to values between 0-255
<br />args:
* image = tensor that contains an image numpy
**denorm_numpy_labels**
<br />description: Reverts your numpy label tensor to be scaled by the width and height args
<br />args:
* labels = tensor that contains an labels numpy
* width = width you'd like to scale by
* height = height you'd like to scale by
