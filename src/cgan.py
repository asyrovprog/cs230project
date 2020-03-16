from keras.layers import Input, Dropout, BatchNormalization, Concatenate, Activation, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import model_from_json, Model
from keras.optimizers import Adam
from src.image_tools import *
import shutil, math
import pandas as pd, sys
from src.cgan_metrics import *
import datetime


class CondGANConfig:
    NAME = "cond_gan"

    # shape of image we condition on
    COND_SHAPE = (256, 256, 3)

    # shape of generated image
    GEN_SHAPE = (256, 256, 3)

    # patch size (see PatchGAN)
    # Note: this constant cannot be simply changed. If we are changing it we must adjust discriminator size at
    # last layer, for instance if we change it to 8, then d5 in discriminator model build code could be uncommented.
    PATCH_SIZE = 16

    # Adam optimizer parameters
    LEARNING_RATE  = 0.0002
    LEARNING_BETA1 = 0.5
    LEARNING_BETA2 = 0.999

    # number of conv filter in generator and discriminators
    GEN_FILTERS  = 64
    DISC_FILTERS = 64

    EPOCHS          = 2
    BATCH_SIZE      = 16
    STATUS_INTERVAL = 200

    ADD_NOISE = True

    GAMMA = 100

    IMAGE_FOLDER = os.path.join("logs", NAME, "images")
    MODEL_FOLDER = os.path.join("logs", NAME, "models")

    # Return batch with id 'bid' of size 'batch_size' [imgs_target, imgs_cond, filenames]
    # values of channels must be re-scaled to range [-1, 1].
    def batch(self, bid, batch_size, is_train = True):
        pass

    # Same format as for batch, except images should be randomly selected
    def random_batch(self, batch_size, is_train = True):
        pass

    # should return number of samples
    def num_samples(self):
        pass


class CondImageGAN:
    def __init__(self, config):
        self.config = config

        self.img_shape  = config.COND_SHAPE
        self.gen_shape  = config.GEN_SHAPE
        self.disc_patch = (config.PATCH_SIZE, config.PATCH_SIZE, 1)

        # Number of filters in the first layer of G and D
        self.gen_filters = config.GEN_FILTERS
        self.disc_filters = config.DISC_FILTERS

        optimizer = Adam(config.LEARNING_RATE, config.LEARNING_BETA1, config.LEARNING_BETA2)

        # Create discriminator computational graph model and compile
        self.discriminator = self.discriminator_graph()
        self.discriminator.compile(loss='binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

        # Creating generator graph model
        self.generator = self.generator_graph()

        # target images (s.t. building footprint) and conditioning images (s.t. satellite images)
        img_target = Input(shape = self.gen_shape)
        img_conditional = Input(shape = self.img_shape)

        # Generator generates target image (s.t. building footprint) based on conditioning
        # image (s.t. satellite images)
        generated_target = self.generator(img_conditional)

        # While training cobined model we keep discriminator weights fixed
        self.discriminator.trainable = False

        # Output of discriminator is validity of each patch
        valid = self.discriminator([generated_target, img_conditional])

        # Combined model
        self.combined = Model(inputs = [img_target, img_conditional], outputs = [valid, generated_target])
        self.combined.compile(loss = ['binary_crossentropy', 'mae'], loss_weights = [1, config.GAMMA], optimizer = optimizer)


    def generator_graph(self):
        # U-Net type generator

        # downsampling
        def conv(X, filters, f_size = 4, batch_norm = True):

            d = Conv2D(filters, kernel_size = f_size, strides = 2, padding = 'same')(X)
            d = LeakyReLU(alpha = 0.2)(d)

            if batch_norm:
                d = BatchNormalization(momentum = 0.8)(d)

            return d

        # upsambling
        def deconv(X, skip_input, filters, f_size = 4, dropout_rate = 0):
            u = UpSampling2D(size = 2)(X)
            u = Conv2D(filters, kernel_size = f_size, strides = 1, padding = 'same', activation = 'relu')(u)

            if dropout_rate:
                u = Dropout(dropout_rate)(u)

            u = BatchNormalization(momentum = 0.8)(u)
            u = Concatenate()([u, skip_input])

            return u

        # Image input
        d0 = Input(shape = self.img_shape)

        # Downsampling
        d1 = conv(d0, self.gen_filters, batch_norm = False)
        d2 = conv(d1, self.gen_filters * 2)
        d3 = conv(d2, self.gen_filters * 4)
        d4 = conv(d3, self.gen_filters * 8)
        d5 = conv(d4, self.gen_filters * 8)
        d6 = conv(d5, self.gen_filters * 8)
        d7 = conv(d6, self.gen_filters * 8)

        # Upsampling
        u1 = deconv(d7, d6, self.gen_filters * 8)
        u2 = deconv(u1, d5, self.gen_filters * 8)
        u3 = deconv(u2, d4, self.gen_filters * 8)
        u4 = deconv(u3, d3, self.gen_filters * 4)
        u5 = deconv(u4, d2, self.gen_filters * 2)
        u6 = deconv(u5, d1, self.gen_filters)

        u7 = UpSampling2D(size = 2)(u6)

        target_channels = self.gen_shape[2]
        generated_image = Conv2D(target_channels, kernel_size = 4, strides = 1, padding = 'same', activation = 'tanh')(u7)

        model = Model(d0, generated_image)

        print("--------------------------------------------------------------------")
        print("Generator Model Summary")
        model.summary()
        print("--------------------------------------------------------------------")
        return model

    def discriminator_graph(self):

        def d_layer(layer_input, filters, f_size = 4, batch_norm = True):

            d = Conv2D(filters, kernel_size = f_size, strides = 2, padding = 'same')(layer_input)
            d = LeakyReLU(alpha = 0.2)(d)

            if batch_norm:
                d = BatchNormalization(momentum = 0.8)(d)

            return d

        img_target = Input(shape = self.gen_shape)
        img_cond = Input(shape = self.img_shape)

        # Concatenate conditioning and target image
        D = Concatenate(axis = -1)([img_target, img_cond])

        if self.config.ADD_NOISE:
            D = GaussianNoise(0.1)(D)

        D = d_layer(D, self.disc_filters, batch_norm = False)
        D = d_layer(D, self.disc_filters * 2)
        D = d_layer(D, self.disc_filters * 4)
        D = d_layer(D, self.disc_filters * 8)
        # D = d_layer(D, self.disc_filters * 8) # D5

        D = Conv2D(1, kernel_size = 5, strides = 1, padding='same', activation = "sigmoid")(D)
        D = Activation(activation = "sigmoid")(D) # patch validity

        model = Model([img_target, img_cond], D)

        print("--------------------------------------------------------------------")
        print("Discriminator Model Summary")
        model.summary()
        print("--------------------------------------------------------------------")
        return model

    def train(self):
        cfg = self.config

        # cleanup log folders
        for p in [cfg.MODEL_FOLDER, cfg.MODEL_FOLDER]:
            if os.path.exists(p):
                shutil.rmtree(p)
                os.makedirs(p, exist_ok = True)

        epochs, batch_size, sample_interval = cfg.EPOCHS, cfg.BATCH_SIZE, cfg.STATUS_INTERVAL

        # Adversarial loss ground truths for actual target images (1s) and generated (0s)
        valid_gt = np.ones((batch_size,) + self.disc_patch) # 1s for each patch
        gen_gt = np.zeros((batch_size,) + self.disc_patch) # 0s for each patch

        ns = cfg.num_samples()
        batch_count = math.ceil(ns / batch_size)
        it = 0
        dl_gt, dl_gen, gl, dacc = [], [], [], []

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}', flush = True)
            s = datetime.datetime.now()

            for bid in range(batch_count):

                # get random batch from dataset
                imgs_target, imgs_cond, _ = cfg.batch(bid, batch_size)

                # Condition on B and generate a translated version
                gen_target = self.generator.predict(imgs_cond)

                # Train discriminator
                disc_loss_real = self.discriminator.train_on_batch(x = [imgs_target, imgs_cond], y = valid_gt)
                disc_loss_gen = self.discriminator.train_on_batch(x = [gen_target, imgs_cond], y = gen_gt)
                disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_gen)

                # Train generator
                g_loss = self.combined.train_on_batch(x = [imgs_target, imgs_cond], y = [valid_gt, imgs_target])

                dl_gt.append(disc_loss_real[0])
                dl_gen.append(disc_loss_gen[0])
                gl.append(g_loss[0])
                dacc.append(disc_loss[1]*100)

                if (it % sample_interval) == 0:
                    self.sample_images(it)

                # progress bar
                p = ((bid + 1)/batch_count) * 100.0
                sys.stdout.write(f"\r[{p:6.2f}%] [{'='*int(p/2)}{' '*(50 - int(p/2))}] DLReal:{disc_loss_real[0]:6.4f}, " +
                                 f"DLGen:{disc_loss_gen[0]:6.4f}, DA:{disc_loss[1]*100:6.2f}, GL:{g_loss[0]:6.4f} " +
                                 f"\t({(datetime.datetime.now() - s).seconds / 60:5.1f} mins)")
                it += 1

            # run model on random batch from validation set
            print()
            _, precision, recall, f1, iou_score = self.evaluate_on_batch(batch_size * 2)
            print(f"\tPrecision: {precision:9.6f}, Recall: {recall:9.6f}, F1: {f1:9.6f}, IoU: {iou_score:9.6f}")

        print()
        self.sample_images(it)
        self.save_model()

        # Save training progress log
        content = {'DLoss_GT': dl_gt, 'DLoss_Gen': dl_gen, 'GLoss': gl, "DAcc": dacc}
        df = pd.DataFrame(content)
        df.to_csv(os.path.join(self.config.MODEL_FOLDER, "training_log.csv"), index = False)

    def evaluate_on_batch(self, batch_size):
        cfg = self.config
        aimg_y, aimg_g = [], []
        for i in range(batch_size):
            img_y, img_x, _ = cfg.random_batch(1, False)
            img_g = self.predict(img_x)
            aimg_y.append(img_y[0])
            aimg_g.append(img_g[0])
        return compute_cgan_metrics_batch(aimg_y, aimg_g)

        # predict target images based on passed conditioning images
    def predict(self, condition, apply_mask = True):
        predictions = self.generator.predict(condition)
        if apply_mask:
            predictions[predictions >= 0] = 1.0
            predictions[predictions < 0] = -1.0
        return predictions


    # load previously saved model
    def load_models(self, weights_only = False):
        cfg = self.config
        modpath = cfg.MODEL_FOLDER

        def load_model(model, name, wo):
            fmodel = os.path.join(modpath, f"{name}.json")
            fweights = os.path.join(modpath, f"{name}_weights.hdf5")

            if not wo:
                json_file = open(fmodel, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)

            model.load_weights(fweights)
            return model

        self.generator = load_model(self.generator, "ccgan_generator", weights_only)
        self.discriminator = load_model(self.discriminator, "ccgan_discriminator", weights_only)


    def sample_images(self, iteration, count = 3):
        """
        Predict (generate images) for 'count' random images from 'dev' dataset for model performance analysis
        """
        img_folder = self.config.IMAGE_FOLDER
        for i in range(count):
            target, condition, files = self.config.random_batch(1, False) # get random images from validation dataset
            generated = self.predict(condition)
            target, condition, generated, file = image2rgb(target[0]), image2rgb(condition[0]), image2rgb(generated[0]), files[0]

            iter_folder = os.path.join(img_folder, f"iter{iteration}")
            os.makedirs(iter_folder, exist_ok = True)

            # extract name of image without extension
            imgfile = os.path.basename(file)
            imgfile = os.path.splitext(imgfile)[0]

            if condition.shape[2] > 3:
                condition = condition[:,:,:3]
            save_image(iter_folder, imgfile + "_input", condition)
            save_image(iter_folder, imgfile + "_target", target)
            save_image(iter_folder, imgfile + "_prediction", generated)

    def save_model(self):
        """
        Save model for further evaluation on dev and test (see experiments.urban3d_validation_cgan.py)
        """
        def save(model, model_name, cfg):
            modpath = cfg.MODEL_FOLDER
            os.makedirs(modpath, exist_ok = True)

            model_path = os.path.join(modpath, f"{model_name}.json")
            weights_path = os.path.join(modpath, f"{model_name}_weights.hdf5")

            options = {"file_arch": model_path, "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)

            model.save_weights(options['file_weight'])

        save(self.generator, "ccgan_generator", self.config)
        save(self.discriminator, "ccgan_discriminator", self.config)

