# segmentationClassification_BrainTumour
Final project in the course Spatial statistics and image analysis.

Medical image analysis is an increasingly important concept for aiding patient
diagnosis. But it is a tedious and time consuming task relying on subjective (albeit
expert) knowledge of clinicians and technicians. Decision support systems relying
on standardised analysis pipelines and classification could greatly aid and speed
up these processes. Brain tumours are one of the more important diseases when it
comes to early classification and treatment selection. In this project I propose an
analysis pipeline for automatic segmentation unhealthy tissue, based on a dataset
of 3064 MR images from tumour patients with 3 different kinds of tumours. The
pipeline consists of three parts: preprocessing, segmentation and classification. The
aim of the project is to see whether automatic segmentation using a residual neural
net can improve classification of extracted features using support vector machines,
compared to classification of features extracted from full images. Results indicate
that the proposed segmentation yields a better accuracy in the classification part
than using full images. However, the improvement is only slight. The small dataset
is a limitation of the study and future research with more data should be performed
to solidify any potential effects.

The proposed pipeline:

![image](https://user-images.githubusercontent.com/62261432/156571262-986ce3e7-fb3c-473b-b6e6-372f4e2d0818.png)

Data must be downloaded from its original [source](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427).

For further details see the [report](https://github.com/jorittmo/segmentationClassification_BrainTumour/blob/master/report/write_up.pdf).
