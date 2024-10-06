### Data Augmentation and model setups ###

dir_path="/content/yeni deneme/Garbage classification/Garbage classification"

train = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         validation_split=0.35,
                         rescale=1./255,
                         shear_range = 0.5,
                         zoom_range = 0.5,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,)

val = ImageDataGenerator(rescale=1/255,
                        validation_split=0.1)



train_generator = train.flow_from_directory(
    dir_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  
    subset='training'
)

validation_generator = val.flow_from_directory(
    dir_path,
    target_size=(224, 224),
    batch_size=32,  
    class_mode='categorical',  
    subset='validation'
)

model = Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(6, activation='softmax')  
])

metrics = [
    "accuracy",
    AUC(name='auc')
]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  
              metrics=metrics)


early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    mode='max',
    verbose=1,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath='garbage_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

start_time = datetime.datetime.now()

history = model.fit(
    train_generator,
    epochs=40,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

end_time = datetime.datetime.now()
total_duration = end_time - start_time
print("Training Time:", total_duration)


val_loss, val_accuracy, val_auc = model.evaluate(validation_generator, verbose=0)
print(f"Loss: {val_loss}")
print(f"Accuracy: {val_accuracy}")
print(f"AUC: {val_auc}")

Loss: 0.9795839786529541
Accuracy: 0.6892430186271667
AUC: 0.9112585186958313



waste_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper',4: 'plastic', 5: 'trash'}
im_dir = "/content/yeni deneme/Garbage classification/Garbage classification"
garbage_model = load_model('/content/garbage_model.keras')
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)

    img_array = image.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array


def prediction_probs(img_array, model, waste_labels):

    predictions = model.predict(img_array, verbose = 0)

    predicted_class_idx = np.argmax(predictions[0])

    predicted_class = waste_labels.get(predicted_class_idx, 'Unknown')

    max_probability = np.max(predictions[0])

    return max_probability, predicted_class
  def display_images(image_paths, model, waste_labels):
    """Display images along with their predicted and true labels."""
    num_images = len(image_paths)
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols
    plt.figure(figsize=(num_cols * 5, num_rows * 5))

    for i, path in enumerate(image_paths):
        img, img_array = preprocess_image(path)

        probability, predicted_class = prediction_probs(img_array, model, waste_labels)

        ax = plt.subplot(num_rows, num_cols, i + 1)
        img = image.img_to_array(img)
        plt.imshow(img.astype('uint8'))

        true_label = path.split('/')[-2]

        plt.title(f"Max Probability: {probability:.2f}\nPredicted Class: {predicted_class}\nTrue Class: {true_label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    get_image_paths(im_dir, 20)



random_images_path = get_image_paths(im_dir, 20)
display_images(random_images_path, garbage_model, waste_labels)


