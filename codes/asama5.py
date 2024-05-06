from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

classes = [17, 22, 53, 55, 60, 71, 90]
class_map = {k: i for i, k in enumerate(classes)}

y_train_mapped = np.array([class_map[y[0]] for y in y_train if y[0] in classes])
y_test_mapped = np.array([class_map[y[0]] for y in y_test if y[0] in classes])

x_train_filtered = np.array([x for i, x in enumerate(x_train) if y_train[i][0] in classes])
x_test_filtered = np.array([x for i, x in enumerate(x_test) if y_test[i][0] in classes])

x_train_last = x_train_filtered.astype('float32') / 255
x_test_last = x_test_filtered.astype('float32') / 255

y_train_categorical = to_categorical(y_train_mapped, num_classes=len(classes))
y_test_categorical = to_categorical(y_test_mapped, num_classes=len(classes))

model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), padding='same'),
    Activation('relu'),
    GlobalAveragePooling2D(),
    
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train_last, y_train_categorical, epochs=100, validation_data=(x_test_last, y_test_categorical), batch_size=50,callbacks=[early_stopping])

loss, accuracy = model.evaluate(x_test_last, y_test_categorical)
print(f'Test Doğruluğu: {accuracy*100:.2f}%')

unique_classes = np.unique(y_test_mapped)
indices_per_class = {class_id: np.where(y_test_mapped == class_id)[0][5] for class_id in unique_classes}

selected_images = np.array([x_test_filtered[indices_per_class[class_id]] for class_id in unique_classes])
selected_labels = np.array([y_test_mapped[indices_per_class[class_id]] for class_id in unique_classes])

predictions = model.predict(selected_images)

fig, axes = plt.subplots(1, len(unique_classes), figsize=(15, 2))
for i, ax in enumerate(axes):
    ax.imshow(selected_images[i].astype('uint8'))
    ax.title.set_text(f'Class: {classes[unique_classes[i]]}\nPred: {np.argmax(predictions[i])}')
    ax.axis('off')

plt.show()

for i, prediction in enumerate(predictions):
    print(f"{classes[unique_classes[i]]}. Sınıf İçin Çıkış Vektörleri:\n{prediction}\n")