from keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = cifar100.load_data(label_mode='fine')

class_names = [
    'castle',
    'clock',
    'orange',
    'otter',
    'plain',
    'sea',
    'train'
]

classes = [17, 22, 53, 55, 60, 71, 90]

fig, axes = plt.subplots(len(classes), 10, figsize=(20, 16))

plt.subplots_adjust(left=0.07   , hspace=0.1, wspace=0.1)

for i, cls in enumerate(classes):
    indices = np.where(train_labels.flatten() == cls)[0]
    random_indices = np.random.choice(indices, 10, replace=False)
    
    y_position = axes[i, 0].get_position().y0 + axes[i, 0].get_position().height / 2
    
    fig.text(0.02, y_position, f'{class_names[i]}', fontsize=14, verticalalignment='center', transform=fig.transFigure)
    
    for j, idx in enumerate(random_indices):
        ax = axes[i, j]
        ax.imshow(train_images[idx], aspect='auto')
        ax.axis('off')

plt.show()