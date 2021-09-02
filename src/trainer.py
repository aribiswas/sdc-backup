import tensorflow as tf
import model as mdl
import dataprocessor as proc
import os
import datetime


timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = "../trained_models"
checkpoint_path = "../checkpoints/" + timestamp_str + "/model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
log_dir = "../logs/fit/" + timestamp_str

input_shape = (224, 224, 3)
num_breeds = 120

ds_train, ds_test, ds_info = proc.load_dataset()
train_batches = proc.prepare(ds_train, input_shape, num_breeds, batch_size=32)
test_batches = proc.prepare(ds_test, input_shape, num_breeds,  batch_size=32)


def create_model():
    # Create a new model instance
    # net = mdl.alexnet(input_shape, num_breeds, lr=0.001)
    # net = mdl.vgg16(input_shape, num_breeds, lr=0.0001)
    # net = mdl.resnet50(input_shape, num_breeds, lr=0.0002)  # 0.01
    net = mdl.mobilenet(input_shape, num_breeds, lr=0.0001)  # 0.01

    return net


def evaluate(net):
    print("Evaluating model...")
    # Evaluate the model
    metrics = net.evaluate(test_batches, return_dict=True, verbose=1)
    for key, val in metrics.items():
        print(key + ": {:.2f}".format(val))


def learn(net):
    print("Training model...")

    # Analyze the model
    net.summary()
    # evaluate(net, test_batches)

    # Create a callback for visualization in TensorBoard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1,
                                                          profile_batch=0)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Train
    net.fit(
        train_batches,
        epochs=10,
        validation_data=test_batches,
        callbacks=[tensorboard_callback]  # , cp_callback]
    )

    # save the model weights
    net.save(save_path + "/mobilenet_1")


def main():
    tf.random.set_seed(0)
    net = create_model()
    learn(net)


if __name__ == "__main__":
    main()
