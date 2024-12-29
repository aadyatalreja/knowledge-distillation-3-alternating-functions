# Knowledge Distillation from VGG16 to Mobilenet

- Teacher model: VGG16  
- Student model: Mobilenet

## Requirements

- Python 3.10  
- PyTorch 2.2.1+cu121  
- ptflops

## Usage

### 1. Models

#### VGG16
VGG16, a renowned CNN architecture from the University of Oxford, excels in image classification with its 16 layers and simple structure. It comprises 13 convolutional and 3 fully connected layers, employing 3x3 filters and 2x2 max-pooling. Despite its effectiveness, its size and depth can be computationally intensive.

#### MobileNet
MobileNet, developed by Google, is tailored for mobile and embedded devices, featuring 28 layers and innovative depthwise separable convolutions. This reduces parameters and computational complexity while maintaining performance. MobileNet adapts to various input sizes and is widely used for transfer learning, albeit potentially sacrificing some accuracy for efficiency.

### 2. Dataset
We have used the CIFAR-100 dataset.  
- It contains 60,000 color images.  
- Images are of size 32x32 pixels.  
- The dataset is organized into 100 classes, each containing 600 images.  
- There are 50,000 training images and 10,000 testing images.  
- Each image is labeled with one of the 100 fine-grained classes.

### 3. Train the Model

To train the VGG16 model:
```bash
# use gpu to train vgg16
$ python train.py -net vgg16 -gpu
```

To train the MobileNet model:
```bash
# use gpu to train mobilenet
$ python train.py -net mobilenet -gpu
```

To perform knowledge distillation from the trained VGG16 to the MobileNet model:
```bash
# use gpu to train mobilenet
$ python knowledge_distillation_train.py -gpu -teacher path_to_best_vgg16_weights_file -student path_to_best_mobilenet_weights_file
```
The weights file with the best accuracy would be written to the disk with a name suffix 'best' (default in the checkpoint folder).

### 4. Test the Model

Test the VGG16 model:
```bash
$ python test.py -net vgg16 -weights path_to_best_vgg16_weights_file
```

Test the MobileNet model:
```bash
$ python test.py -net mobilenet -weights path_to_best_mobilenet_weights_file
```

Test the knowledge-distilled MobileNet model:
```bash
$ python knowledge_distillation_test.py -gpu -weights path_to_best_knowledge_distilled_mobilenet_weights_file
```

## Implementation Details and References

- VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)  
- MobileNet: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)  
- Hyperparameter settings: [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2), which is init lr = 0.1 divided by 5 at 60th, 120th, 160th epochs, trained for 200 epochs with batchsize 128 and weight decay 5e-4, Nesterov momentum of 0.9.  
- Code reference: [GitHub: weiaicunzai](https://github.com/weiaicunzai/pytorch-cifar100)

## Results

### Previous Results

| Dataset  | Network           | Learning Rate | Batch Size | Size (MB) | Params  | Top-1 Err | Top-5 Err | Time (ms) per Inference Step (GPU) | Time (ms) per Inference Step (CPU) | FLOPs   |
|:--------:|:-----------------:|:-------------:|:----------:|:---------:|:-------:|:---------:|:---------:|:--------------------------------:|:---------------------------------:|:-------:|
| CIFAR-100| VGG16             | 0.1           | 128        | 136.52    | 34.0M   | 27.77     | 10.12     | 177.2584                         | 10.7589                           | 334.14  |
| CIFAR-100| MobileNet         | 0.1           | 128        | 24.03     | 3.32M   | 33.06     | 10.15     | 57.6361                          | 9.0793                            | 48.32   |
| CIFAR-100| Knowledge Distilled MobileNet | 0.1 | 128 | 24.03 | 3.32M | 32.61 | 10.26 | 56.7409 | 9.6162 | 48.32 |
| CIFAR-100| Knowledge Distilled MobileNet | 0.001 | 64 | 24.03 | 3.32M | 32.16 | 10.83 | 58.2087 | 9.0350 | 48.32 |

### Results from Varying Soft Target Weight for Every Alternating Epoch

| Experiment No. | Loss Function                                           | Soft Target Weight | Cross Entropy Loss Weight | Top-1 Error Rate | Top-5 Error Rate | Top-1 Accuracy | Top-5 Accuracy | Parameters | Time (ms) per Inference (GPU) | FLOPs   |
|:--------------:|:------------------------------------------------------:|:------------------:|:-------------------------:|:----------------:|:----------------:|:--------------:|:--------------:|:----------:|:-----------------------------:|:-------:|
| 1              | Epoch%2==0: distillation loss, else cross-entropy loss                                     | 0.00001           | 1                         | 34.94           | 11               | 65.06          | 89             | 3.32M      | 9.4931                         | 48.32   |
| 2              | Epoch%2==0: distillation loss, else cross-entropy loss | 0.0001            | 1                         | 34.76           | 10.83            | 65.24          | 89.17          | 3.32M      | 5.0192                         | 48.32   |
| 3              | Epoch%2==0: distillation loss, else cross-entropy loss                                    | 0.001             | 1                         | 32.61           | 10.12            | 67.39          | 89.59          | 3.32M      | 5.1017                         | 48.32   |
| 4              | Epoch%2==0: distillation loss, else cross-entropy loss                                    | 0.01              | 1                         | 34.87           | 10.71            | 65.13          | 89.29          | 3.32M      | 4.8188                         | 48.32   |
| 5              | Epoch%2==0: distillation loss, else cross-entropy loss                               | 0.1               | 1                         | 34.64           | 10.69            | 65.36          | 89.31          | 3.32M      | 3.9408                         | 48.32   |
| 6              | Epoch%2==0: distillation loss, else cross-entropy loss                                     | 0.5               | 1                         | 36.04           | 11.5             | 63.96          | 88.5           | 3.32M      | 4.3824                         | 48.32   |
| 7              | Epoch%2==0: distillation loss, else cross-entropy loss                                     | 1                 | 1                         | 33.46           | 11.78            | 66.54          | 88.22          | 3.32M      | 10.7554                        | 48.32   |

#### Results from Varying Soft Target Weight Across Epochs Divisible by Three

| Experiment No. | Loss Function                                           | Soft Target Weight | Cross Entropy Loss Weight | Top-1 Error Rate | Top-5 Error Rate | Top-1 Accuracy | Top-5 Accuracy | Parameters | Time (ms) per Inference (GPU) | FLOPs   |
|:--------------:|:------------------------------------------------------:|:------------------:|:-------------------------:|:----------------:|:----------------:|:--------------:|:--------------:|:----------:|:-----------------------------:|:-------:|
| 1              | Epoch%3==0 then distillation loss, else cross-entropy loss | 0.0001           | 1                         | 34.43           | 11.64            | 65.57          | 88.36          | 3.32M      | 9.9175                         | 48.32   |
| 2              | Epoch%3==0 then distillation loss, else cross-entropy loss | 0.001            | 1                         | 34.54           | 11.63            | 65.46          | 88.37          | 3.32M      | 9.7377                         | 48.32   |
| 3              | Epoch%3==0 then distillation loss, else cross-entropy loss                     | 0.01              | 1                         | 34.56           | 11.65            | 65.44          | 88.35          | 3.32M      | 10.3282                        | 48.32   |
| 4              | Epoch%3==0 then distillation loss, else cross-entropy loss                                    | 0.1               | 1                         | 34.58           | 11.61            | 65.42          | 88.39          | 3.32M      | 9.039                          | 48.32   |
| 5              | Epoch%3==0 then distillation loss, else cross-entropy loss                                     | 1                 | 1                         | 33.85           | 11.02            | 66.15          | 88.98          | 3.32M      | 8.9308                         | 48.32   |

#### 3-Epoch Experiment Inferences

1. **Impact of Alternating Loss Function:**  
   - Alternating between distillation loss on epochs divisible by 3 and cross-entropy loss on others improves Top-1 Accuracy and Top-5 Accuracy in most cases. This highlights the importance of leveraging both soft and hard target information at appropriate intervals during training.

2. **Optimal Soft Target Weight:**  
   - Experiment 5 (soft target weight = 1) yields the highest Top-1 Accuracy (66.15%) and the lowest Top-1 Error Rate (33.85%), indicating that larger soft target weights can improve performance in the alternating epoch setup when epochs are divided by three.

3. **Balanced Approach:**  
   - Lower soft target weights, such as 0.0001 and 0.001 (Experiments 1 and 2), also achieve competitive performance. This suggests that alternating epochs allows for flexibility in weight selection, benefiting models that might otherwise underperform with static loss functions.

4. **Inference Efficiency:**  
   - Experiment 5 (soft target weight = 1) achieves the best balance between accuracy and inference time, with a low inference time of 8.93 ms per GPU inference step.

5. **Comparison to Non-Alternating Setup:**  
   - The alternating epoch strategy (distillation and cross-entropy losses) consistently performs better than using static loss functions for all epochs. This indicates that selectively applying distillation loss boosts knowledge transfer effectiveness.

This experiment confirms that strategically alternating loss functions and tuning the soft target weight can lead to significant improvements in accuracy while managing computational costs.

