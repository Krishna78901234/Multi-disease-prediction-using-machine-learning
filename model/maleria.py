import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the VGG19 base model with pre-trained weights
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top of the base model
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(2, activation='softmax')(x)  # Adjust output neurons to match your classes (2 classes in this case)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare the data
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\prava\\Desktop\\multi disease prediction\\multi disease prediction\\dataset\\maleriadata\\Train', 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_generator = val_datagen.flow_from_directory(
    'C:\\Users\\prava\\Desktop\\multi disease prediction\\multi disease prediction\\dataset\\maleriadata\\Test', 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=5)

# Save the model
model.save('C:\\Users\\prava\\Desktop\\multi disease prediction\\multi disease prediction\\model\\modelvgg19.h5')
print("Model saved as modelvgg19.h5")
 