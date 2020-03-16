from keras.callbacks import TensorBoard
from time import time
import imgaug
import os
from imgaug import augmenters as iaa

def calculate_layers(config, value):
    updated_layer_regex = {
        # all layers but the backbone
        "heads": r"(conv1)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        # From a specific Resnet stage and up
        "3+": r"(conv1)|(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        "4+": r"(conv1)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        "5+": r"(conv1)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        # All layers
        "all": ".*",
    }
    l = value
    if config.IMAGE_TYPE.lower() == "rgb" or not l in updated_layer_regex:
        return l
    else:
        return updated_layer_regex[l]

def multi_train(config, dataset_train, dataset_val, model):
    """
    Training in sevelar steps goting deeper and deeper. This helps when we already have pre-trained model. We start
    from head, train it, and then, redusing learning rate, train more layer, optionally, as final step,
    we can train entire model.
    """

    if config.EXT_USE_AUGMENTATION:

        # https://github.com/aleju/imgaug
        print("INFO: Image augmentation is enabled")

        affine = iaa.Affine(
            scale={"x": (1.0, 1.2), "y": (1.0, 1.2)}, # scale images to 100-120% of their size, individually per axis
            rotate=(-45, 45) # rotate by -45 to +45 degrees
            )

        augmentation = iaa.Sometimes(0.7, [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            affine])
    else:
        augmentation = None

    tensorboards = []

    if config.LEARNING_RATES is None:
        config.LEARNING_RATES = [config.LEARNING_RATE]
        config.LEARNING_LAYERS = ["heads"]
        config.LEARNING_EPOCHS = [16]

    for i in range(len(config.LEARNING_RATES)):

        tdir = os.path.join(model.log_dir, f"step{i}")
        tensorboard = TensorBoard(log_dir = tdir, update_freq = 'batch', write_images = True)
        tensorboards.append(tdir)

        model.train(dataset_train, dataset_val,
                    learning_rate = config.LEARNING_RATES[i],
                    epochs = config.LEARNING_EPOCHS[i],
                    augmentation = augmentation,
                    custom_callbacks = [tensorboard],
                    layers = calculate_layers(config, config.LEARNING_LAYERS[i]))

    print(f"INFO: Tensorboard(s):")
    for tdir in tensorboards:
        print(f"\t'tensorboard --logdir={tdir}/'")