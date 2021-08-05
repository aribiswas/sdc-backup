import tensorflow as tf
import model as mdl
import dataprocessor as proc

# Load data from stanford dogs dataset
ds_train, ds_test, ds_info = proc.load_dataset()

# Build a training pipeline
ds_train = ds_train.map(proc.preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# Build evaluation pipeline
ds_test = ds_test.map(proc.preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# Create a model
basic_model = mdl.basicnet((120, 120, 3))

# Train
basic_model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test
)


