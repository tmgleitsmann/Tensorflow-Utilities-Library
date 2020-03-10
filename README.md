# Tensorflow-Utilities-Library
Most of Tensorflow's APIs are incredibly low level with horrendous documentation. This personal library will be used to solve a lot of common Tensorflow problems.


## Image
### Directory that handles working with image data.
**csv_to_dataset**
<br />description: Load in CSV data directly into a tensorflow dataset object. 
<br />CSV data must be in the format : <br />
1. First column is ignored. Class assumes column indexes images & labels numerically.
2. Second column is path to images. Needs to be path/image.jpg .
3. Third column are the labels. Labels are unchanged during the creation process.
<br />args:
* csv_path = posix path to CSV file
* batch_size = number : used to create dataset objects
* width = number : used to reshape image width
* height = number : used to reshape image height
* train_test_split = number : used to allocate data to training & validation
* aug_flag = boolean : used to include/ignore image augmentation during pre-processing

*functions*
<br />
* **dunder call overridden:** Can now call on the class itself to retrieve dataset & validation set as long as class has been properly initialized.
