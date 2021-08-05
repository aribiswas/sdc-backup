import tensorflow as tf
import model as mdl
import dataprocessor as proc
import os


checkpoint_path = "../checkpoints/train_1.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


def train_model():
    # Load data from stanford dogs dataset
    ds_train, ds_test, ds_info = proc.load_dataset()

    # Build a training pipeline
    ds_train = ds_train.map(lambda im, label: proc.preprocess_img(im, label, (120, 120)),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Build evaluation pipeline
    ds_test = ds_test.map(lambda im, label: proc.preprocess_img(im, label, (120, 120)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    # Create a model
    net = mdl.alexnet(input_size=(120, 120, 3))
    net.summary()
    evaluate_model(net, ds_test)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Train
    net.fit(
        ds_train,
        epochs=10,
        validation_data=ds_test,
        callbacks=[cp_callback]
    )

    evaluate_model(net, ds_test)


def evaluate_model(net, test_ds):
    # Evaluate the model
    loss, acc = net.evaluate(test_ds, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


def load_model():
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # Create a new model instance
    net = mdl.alexnet(input_size=(120, 120, 3))

    # Load the previously saved weights
    net.load_weights(latest)

    return net


train_model()


