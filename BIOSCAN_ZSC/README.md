# BIOSCAN-5M

![Alt Text](bioscan_zsc.png)

###### <h3> Overview
The zero-shot clustering tasks in our benchmark experiments unfold in two steps

- (1) The pretrained encoders, trained with either self-supervised learning (SSL) methods like DINO and MAE, or a 
supervised learning method such as cross-entropy, are utilized. 
It's worth noting that our supervised pretraining is conducted on the ImageNet-1k dataset; hence, 
the BIOSCAN-5M Insect dataset is not involved in the pretraining procedure. 
- (2) The representation vectors embedded by all encoders are clustered using various clustering methods. 
Finally, we assess the clusterability of SSL methods and supervised learning using adjusted mutual information (AMI)as well as silhouette score.