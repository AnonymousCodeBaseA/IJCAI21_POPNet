# IJCAI21_POPNet

Source code for the under-review IJCAI2021 paper, ***POPNet: Improved Few-Shot Learning on Android Malware Encrypted Traffic Classification***

The project is written in Pytorch using the following backend: Intel i7-9750 @2.6GHz, 16GB RAM, NVIDIA GeForce RTX2060; Windows 10, CUDA 10.1, and PyTorch 1.0.1.

## Data



Two datasets are Involved, including  MalDroid2017 and USTC2016.

**MalDroid2017** dataset. Download from https://www.unb.ca/cic/datasets/andmal2017.html

**USTC2016** dataset. Download from https://github.com/yungshenglu/USTC-TFC2016

Toolkits are available at https://github.com/yungshenglu/USTC-TFC2016



Data folders are organized as follows:



```
train/
    class1/
        image1.png
        image2.png
        ...
    class2/
        image1.png
        image2.png
        ...
    ...
val/
	./
test/
	./
backup_train/
	aux_class1/
		image1.png
		image2.png
        ...
    aux_class2/
		image1.png
		image2.png
        ...
    ...
novel-backup/
	./  
val-origin/
    class1/
        image1.png
        image2.png
        ...
    class2/
        image1.png
        image2.png
        ...
    ...
    
val-ustc/
    class1/
        image1.png
        image2.png
        ...
    class2/
        image1.png
        image2.png
        ...
    ...
```



Here, 42 classes of target data are stored in ```train```.  Aux classes are stored in ```backup_train```. Testing data of 42 classes of target data are stored in ```val-origin```.  USTC2016 for novelty testing is stored in ```val-ustc```.



## Evaluation



#### Few-shot Android Malware Encrypted Traffic Detection with Large-scale Classes

```python
# without auxiliary
python ./src_or/train.py  --cuda --classes_per_it_val 42   --epochs 100 --large_scale True

# with auxiliary
python ./src_aux/train.py  --cuda --classes_per_it_val 42   --epochs 100 --large_scale True
```



#### Zero-shot Learning on Cross-Platform Malware Traffic Detection

```python
# without auxiliary
python ./src_or/train.py  --cuda --classes_per_it_val 10   --epochs 100 --novelty_dec True

# with auxiliary
python ./src_aux/train.py  --cuda --classes_per_it_val 10   --epochs 100 --novelty_dec True
```



## Results

We compare our methods with state-of-the-art conventional detectors, including FlowPrint,  AppScanner, Datanet,   ACGAN, ACNN, LSTM, RBRN and FC-Net. In addition, state-of-the-art few-shot learning approaches are considered as baseline, including Prototypical Nets, Matching Nets, Triplet Nets, 5-Tuple Nets, MAML,  GNN and Relation Nets.



- Experimental results evaluated on MalDroid2017. Performance of conventional detectors classifying on 42 classes DroidMal traffic with 20 samples per class.

| Methods        | OA        | Pr        | Re        | F1        |
| -------------- | --------- | --------- | --------- | --------- |
| Datanet        | 6.51      | 9.18      | 6.95      | 7.91      |
| ACGAN          | 6.75      | 6.87      | 6.77      | 6.82      |
| LSTM           | 7.01      | 6.79      | 7.02      | 6.91      |
| ACNN           | 1.96      | 2.35      | 2.74      | 2.53      |
| Flowprint      | 14.67     | 15.38     | 14.67     | 15.02     |
| Appscanner     | 1.26      | 34.64     | 1.26      | 2.43      |
| RBRN           | 29.26     | 34.65     | 29.39     | 31.80     |
| FC-Net         | 56.25     | 57.90     | 56.13     | 57.00     |
| **POPNet**     | **66.29** | **66.57** | **69.16** | **67.85** |
| **POPNet-Aux** | **77.44** | **77.09** | **77.10** | **77.10** |



- Experimental results evaluated on MalDroid2017. Few-shot-based methods are estimated with OA with different ***N*** when training. ***N*** is set to 42 while testing. In addition, the auxiliary classes are included in few-shot-based methods, respectively.

| Methods               | 5w        | 10w       | 20w       | 40w       |
| --------------------- | --------- | --------- | --------- | --------- |
| Prototypical Nets     | 42.25     | 40.87     | 38.17     | 36.27     |
| Matching Nets         | 35.74     | 35.58     | 32.94     | 21.95     |
| MAML                  | 4.01      | 5.54      | 6.33      | 7.65      |
| GNN                   | 18.20     | 10.19     | 9.84      | 5.45      |
| Relation Nets         | 20.1      | 10.04     | 7.89      | 6.91      |
| Triplet Nets          | 23.08     | 23.19     | 26.50     | 29.89     |
| 5-Tuple Nets          | 22.71     | 23.02     | 27.26     | 29.71     |
| Prototypical Nets-Aux | 63.14     | 66.16     | 66.87     | 68.14     |
| Matching Nets-Aux     | 41.08     | 42.58     | 40.28     | 43.03     |
| MAML-Aux              | 4.03      | 5.46      | 6.09      | 7.15      |
| Triplet Nets-Aux      | 26.29     | 28.65     | 32.54     | 33.78     |
| 5-Tuple Nets-Aux      | 33.90     | 36.59     | 37.12     | 38.23     |
| **POPNet**            | **64.43** | **66.29** | **64.19** | **60.20** |
| **POPNet-Aux**        | **74.89** | **75.45** | **75.60** | **77.44** |



- Ablation study of components. Prototype  Optimization and Prototype-Vectorized Optimization  are equipped respectively and report the results of OA. Experiments are evaluated on regenerated MalDroid2017 where ***N*** ranges from 5 to 40 when training and is set to 42 during testing.

| Without Auxiliary       |  5w        |  10w       |  20w       |  40w       |
| --------------------- | --------- | --------- | --------- | --------- |
| Prototypical Nets     |42.25|40.87|38.17|36.27|
| POPNet without PO     |43.81|41.31|38.57|38.63|
| POPNet without TO     |61.53|63.41|60.38|59.48|
| **POPNet**          | **64.43** | **66.29** | **64.19** | **60.20** |



| With Auxiliary | 5w | 10w | 20w | 40w |
| ----------------- | ----------- | ------------ | ------------ | ------------ |
| Prototypical Nets | 63.14       | 66.16        | 66.87        | 68.14        |
| POPNet without PO | 57.88       | 65.15        | 67.17        | 68.29        |
| POPNet without TO | 72.28       | 74.24        | 74.97        | 76.68        |
| **POPNet**        | **74.89**      | **75.45**       | **75.60**       | **77.44**       |



- Further validation on Omniglot. One hundred classes are randomly picked up. A 5-Way-10-Shot strategy is employed during trianing while testing with 100-Way-10-shot.

| Omniglot          | Overall Accuracy |
| ----------------- | ---------------- |
| Prototypical Nets | 76.56            |
| Matching Nets     | 52.37            |
| MAML              | 6.27             |
| GNN               | 5.11             |
| Relation Nets     | 21.19            |
| Triplet Nets      | 24.23            |
| 5-Tuple Nets      | 26.23            |
| **POPNet**        | **80.75**        |

Omniglot is available at https://github.com/brendenlake/omniglot/tree/master/python.

Reconstruction code for baseline will be available soon.
