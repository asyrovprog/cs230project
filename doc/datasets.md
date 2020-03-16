# How to download and preprocess datasets

### Download and Install Amazon AWS CLI tools

Process of installing installing aws utility, which is used to download datasets is described here: https://aws.amazon.com/cli/. We need up to 40GB for download entire dataset.

When installed and configured run the following to confirm that we see aws dataset folders:

`aws s3 ls s3://spacenet-dataset/`

### Example of dataset download commands

*Locate files to download. For instance:*

`aws s3 ls s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/`

*Here is example of output:*

```
2019-11-05 11:56:34 1079959064 SN1_buildings_test_AOI_1_Rio_3band.tar.gz
2019-11-05 11:56:15  372882359 SN1_buildings_test_AOI_1_Rio_8band.tar.gz
2019-11-05 11:52:56 2571888708 SN1_buildings_train_AOI_1_Rio_3band.tar.gz
2019-11-05 11:53:36  893706368 SN1_buildings_train_AOI_1_Rio_8band.tar.gz
2019-11-05 11:54:09   21118272 SN1_buildings_train_AOI_1_Rio_geojson_buildings.tar.gz
2019-11-05 11:54:45   77789991 SN1_buildings_train_AOI_1_Rio_metadata.tar.gz
```

*Example of download of single archive:*

`aws s3 cp s3://spacenet-dataset/spacenet/SN4_buildings/train/tarballs/Atlanta_nadir10_catid_1030010003993E00.tar.gz .`

#### List of current datasets location

Here is the list of datasets which we may use for the project. Ideally we should use Urban 3D Challange dataset and at least one other.

1. Urban 3D Challange: `s3://spacenet-dataset/Hosted-Datasets/Urban_3D_Challenge/`
2. SpaceNet Buildings Dataset 1: `s3://spacenet-dataset/spacenet/SN1_buildings/`
3. SpaceNet Buildings Dataset 2: `s3://spacenet-dataset/spacenet/SN2_buildings/`
4. SpaceNet Buildings Dataset 4: `s3://spacenet-dataset/spacenet/SN4_buildings/`

### Creating Training, Validation and Test Datasets

1. Setup environment variable DATASET_ROOT and make sure it exists. 
2. Run 













