import os
import time
import math
import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

constants = {
    "image_size": (28, 28),
    "n_channels": 1,
    "batch_size": 300,
    "num_examples": 9,
    "norm_factor": 255. / 2,
    "latent_dims": 100,
    "epochs": 70,
}

paths = {
    "dataset": "dataset",
    "checkpoint": "training-checkpoints",
    "generated_images": "generated-images",
    "models": "models",
}


class DataLoader():
    def __init__(self, image_size=(64, 64), batch_size=32, norm_factor=None):
        self.image_size = image_size
        self.batch_size = batch_size
        self.norm_factor = norm_factor

    # def load(self, path):
    #     ds = keras.utils.image_dataset_from_directory(
    #         directory=path,
    #         image_size=self.image_size,
    #         seed=101
    #     )\
    #         .unbatch()\
    #         .filter(lambda x, y: tf.equal(y, 2) or tf.equal(y, 3))\
    #         .batch(self.batch_size)

    #     ds = ds.shuffle(buffer_size=5000)\
    #         .map(self.preprocess)\
    #         .prefetch(1)\
    #         .cache()

    #     return ds

    def load(self, path):
        (ds, _), _ = keras.datasets.fashion_mnist.load_data()
        ds = self.preprocess(ds)
        ds = tf.data.Dataset.from_tensor_slices(ds)\
            .shuffle(60000)\
            .batch(self.batch_size)\
            .prefetch(1)\
            .cache()
        return ds

    # def preprocess(self, data, label):
    #     data = tf.image.resize(data, self.image_size)
    #     return ((data - self.norm_factor) / self.norm_factor, label)

    def preprocess(self, data):
        data = data[..., tf.newaxis]
        return (data - self.norm_factor) / self.norm_factor

    def deprocess(self, data):
        return data * self.norm_factor + self.norm_factor


class GAN():
    def __init__(self, latent_dims, input_shape, batch_size, deprocessor, checkpoint_dir="checkpoints", generated_images_dir="generated", training_size=None):
        self.latent_dims = latent_dims
        self.image_shape = input_shape
        self.batch_size = batch_size
        self.training_size = training_size
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.deprocessor = deprocessor
        self.generated_images_dir = generated_images_dir

        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = keras.optimizers.Adam(1e-4)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        model = keras.Sequential([
            keras.layers.Dense(7 * 7 * 256, input_shape=(self.latent_dims, )),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Reshape((7, 7, 256)),
            keras.layers.Conv2DTranspose(
                256,
                (4, 4),
                strides=2,
                padding='same',
                use_bias=False
            ),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2DTranspose(
                128,
                (4, 4),
                strides=1,
                padding='same',
                use_bias=False
            ),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2DTranspose(
                64,
                (4, 4),
                strides=2,
                padding='same',
                use_bias=False
            ),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2DTranspose(
                64,
                (4, 4),
                strides=1,
                padding='same',
                use_bias=False
            ),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2DTranspose(
                1,
                (4, 4),
                strides=1,
                padding='same',
                use_bias=False,
                activation='tanh'
            )
        ])
        return model

    def build_discriminator(self):
        model = keras.Sequential([
            keras.layers.Conv2D(
                64,
                (4, 4),
                strides=1,
                padding='same',
                input_shape=self.image_shape
            ),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(64, (4, 4), strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Conv2D(128, (4, 4), strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Conv2D(256, (4, 4), strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Flatten(),
            keras.layers.Dense(1)
        ])
        return model

    def generator_loss(self, fakes):
        return self.cross_entropy(tf.ones_like(fakes), fakes)

    def discriminator_loss(self, reals, fakes):
        real_loss = self.cross_entropy(tf.ones_like(reals), reals)
        fake_loss = self.cross_entropy(tf.zeros_like(fakes), fakes)
        return real_loss + fake_loss

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal(shape=(self.batch_size, self.latent_dims))

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            generator_loss = self.generator_loss(fakes=fake_output)
            discriminator_loss = self.discriminator_loss(
                reals=real_output, fakes=fake_output)

        generator_gradients = generator_tape.gradient(
            generator_loss, self.generator.trainable_variables)
        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

    def get_checkpoint_callback(self):
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        return checkpoint

    def train(self, dataset, epochs, seed, load_from_checkpoint=False):
        checkpointer = self.get_checkpoint_callback()
        if load_from_checkpoint:
            try:
                checkpointer.restore(
                    tf.train.latest_checkpoint(self.checkpoint_dir))
            except:
                print('Could not load from checkpoint')
        start_training = time.time()
        progress = tqdm(range(1, epochs + 1), desc='Epochs')
        for epoch in progress:
            for batch_idx, image_batch in enumerate(dataset):
                self.train_step(image_batch)
                self.generate_and_save(epoch, seed)
                progress.set_description(
                    f'Epoch: {epoch}/{epochs} | Batch: {batch_idx + 1}/{math.ceil(self.training_size/self.batch_size)}'
                )
            if (epoch + 1) % 15 == 0:
                checkpointer.save(file_prefix=self.checkpoint_prefix)
        end_training = time.time()
        print(
            f'Training duration: {math.ceil(end_training - start_training)} seconds')
        self.generate_and_save(epochs, seed)
        return True

    def generate_and_save(self, epoch, test):
        generated_images = self.generator(test, training=False)
        plt.figure(figsize=(4, 4), dpi=200)
        for i in range(generated_images.shape[0]):
            plt.subplot(3, 3, i+1)
            plt.imshow(self.deprocessor(
                generated_images[i].numpy()).astype(np.uint8), cmap='gray')
            plt.axis('off')
        plt.savefig(f'{self.generated_images_dir}/image_at_epoch__{epoch}.png')
        plt.close()


def generate_result(path='result.gif', filenames=[]):
    with imageio.get_writer(path, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def main():
    print('...Starting\n')
    print('......Setting up\n')
    if not os.path.isdir(paths["generated_images"]):
        os.mkdir(paths["generated_images"])
    if not os.path.isdir(paths["checkpoint"]):
        os.mkdir(paths["checkpoint"])
    if not os.path.isdir(paths["models"]):
        os.mkdir(paths["models"])
    print('......Created required directories\n')
    print('......Initializing DataLoader\n')
    dataloader = DataLoader(
        image_size=constants["image_size"],
        batch_size=constants["batch_size"],
        norm_factor=constants['norm_factor'])
    print('......Loading dataset\n')
    dataset = dataloader.load(path=paths["dataset"])
    print(f'......Dataset: {dataset}\n')
    print('.'*80)
    print('\n')
    print('......Generating random seed\n')
    seed = tf.random.normal(
        [constants["num_examples"], constants["latent_dims"]])
    print('......Initializing GAN\n')
    gan = GAN(
        latent_dims=constants["latent_dims"],
        input_shape=constants["image_size"] + (constants["n_channels"], ),
        batch_size=constants["batch_size"],
        deprocessor=dataloader.deprocess,
        checkpoint_dir=paths['checkpoint'],
        generated_images_dir=paths['generated_images'],
        training_size=60000
    )
    print('......GAN generator\n')
    print(gan.generator.summary())
    print('.'*80)
    print('\n')
    print('......GAN discriminator\n')
    print(gan.discriminator.summary())
    print('.'*80)
    print('\n')
    print('......Training GAN\n')
    gan.train(dataset=dataset,
              epochs=constants['epochs'], seed=seed, load_from_checkpoint=True)
    print('......Saving models\n')
    gan.generator.save(os.path.join(paths['models'], 'gan_generator'))
    gan.discriminator.save(os.path.join(
        paths['models'], 'gan_discriminator'))
    print('......Generating GIF\n')
    filenames = sorted(glob.glob(f'{paths["generated_images"]}/image*.png'))
    generate_result(path="result.gif", filenames=filenames)
    print('...Ending\n')
    return True


if __name__ == "__main__":
    main()
