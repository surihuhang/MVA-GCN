# MVA-GCN
MVA-GCN: a machine learning algorithm prediction for dense granule proteins for Apicomplexa

Citation: Lu Z X, Hu H, et al. Citation: Lu Z X, Hu H, et al. Development and validation of a machine learning algorithm prediction for dense granule proteins for Apicomplexa.2023.


## Datasets

* data.zip: the feature matrix of the descriptors, which is calculated based on the protein sequences
* graph.zip: the graph structure data of the descriptor, which is built using K-nearest neighbor graph


# Run steps

1. Run main.py to train the model and obtain result_Derangement.csv which is predicted scores for GRAs.

2. Run Sort.py to obtain the GRAs in order of predicted score from highest to lowest


# Requirements

* SVM is implemented to work under Python 3.9. 
* numpy==1.22.4
* tqdm==4.64.0
* sklearn==0.24.2
* pandas==1.3.4

## Contact

- Please feel free to contact us if you need any help: [hanghu@stu.ahau.edu.cn]
- __Attention__: Only real name emails will be replied. Please provide as much detail as possible about the problem you are experiencing.
