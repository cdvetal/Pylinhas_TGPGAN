import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pylab
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from keras import layers
from main import *
from keras.backend import sigmoid
from PIL import Image

delimiter = os.path.sep

gen_image_cnt = 0
fake_image_cnt = 0
real_fid_path_name = "real_fid_batch"
gen_fid_path_name = "gen_dif_batch"

dpi = 96

epoch_atual = 0
step_atual = 0


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


class dcgan(object):

    def __init__(self,
                 batch_size=32,
                 gens_per_batch=100,
                 classes_to_train=None,
                 run_from_last_pop=True,
                 linear_gens_per_batch=False,
                 log_losses=True,
                 seed=202020212022,
                 run_folder = None,
                 log_digits_class=True,
                 prefix=None):

        self.seed = seed

        self.img_rows = SIZE_IMAGE
        self.img_cols = SIZE_IMAGE
        self.channels = 1
        #self.last_axis = 4  # accounting for population list, 3 for grayscale, 4 for RGB
        self.input_shape = [self.img_rows, self.img_cols, self.channels]
        self.fid_gen_images = None

        self.log_losses = log_losses
        self.log_digits_class = log_digits_class

        self.run_from_last_pop = run_from_last_pop
        self.linear_gens_per_batch = linear_gens_per_batch

        self.batch_size = batch_size
        self.gens_per_batch = gens_per_batch
        self.last_gen_imgs = []
        self.last_pop = None
        self.last_strat = None

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.discriminator = self.make_discriminator_model()
        self.disc_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)

        # paths
        self.run_folder = run_folder if run_folder is not None else prefix + "_" + set_experiment_ID_run()
        self.run_dir = os.getcwd() + delimiter + "runs" + delimiter + prefix + delimiter + self.run_folder + delimiter
        self.gan_images = self.run_dir + "dcgan_images" + delimiter
        self.gallery_res = [1024, 1024, 3]
        self.best_im_dir = self.gan_images + delimiter + "bests" + delimiter
        self.best_for_generation = self.gan_images + delimiter + "best_for_generation" + delimiter
        self.log_dir = self.run_dir + "logs" + delimiter
        self.images_to_evaluate = self.gan_images + "images_to_evaluate" + delimiter

        os.makedirs(self.gan_images, exist_ok=True)
        os.makedirs(self.best_im_dir, exist_ok=True)
        os.makedirs(self.best_for_generation, exist_ok=True)
        os.makedirs(self.images_to_evaluate, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.gloss = 0
        self.dloss = 0
        self.training_time = 0
        self.loss_hist = []

        # sieve the classes
        self.classes_to_train = classes_to_train if classes_to_train is not None else [i for i in range(10)]


        # LOAD DATASET - MNIST
        (self.x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        y_train = y_train.flatten()
        train_mask = np.isin(y_train, self.classes_to_train)

        self.x_train = self.x_train[train_mask]
        self.x_train = self.x_train[:64] # testing purposes

        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, self.channels).astype(
            'float32')
        self.x_train = (self.x_train - 127.5) / 127.5  # Normalize the images to [-1, 1]
        print("Len of selected dataset: ", len(self.x_train))
        print(self.x_train.shape)

        self.x_train = tf.data.Dataset.from_tensor_slices(self.x_train).shuffle(len(self.x_train)).batch(
            self.batch_size)


    def disc_forward_pass(self, **kwargs):
        tensors = kwargs.get('tensors')

        fitnesses = []
        fit_array = self.discriminator(np.array(np.expand_dims(tensors, axis=3)), training=False)

        # scores
        for index in range(len(tensors)):
            fit =float(fit_array[index][0])
            sigmoid_fit = sigmoid(fit).numpy()
            fitnesses.append(sigmoid_fit)

        return fitnesses


    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        # normal
        print(self.input_shape)
        model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=self.input_shape))
        model.add(layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        # classifier
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(1))

        return model

    def compute_losses(self, gen_output, real_output):
        gen_loss = self.cross_entropy(tf.zeros_like(gen_output), gen_output)
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        self.dloss = gen_loss + real_loss
        self.gloss = -self.dloss
        self.loss_hist.append([self.dloss.numpy(), self.gloss.numpy()])

    def print_training_hist(self):
        for h in self.loss_hist:
            print(h)

    def train_step(self, images, step, epoch):
        global gen_image_cnt, fake_image_cnt
        fake_image_cnt += len(images)



        with tf.GradientTape() as disc_tape:
            ep = self.gens_per_batch if self.linear_gens_per_batch else round(step / 10) + 5  #self.linear_gens_per_batch: TRUE OR FALSE para ter um nr fixo de geracoes ou nao (true nr fixo)
            gen_image_cnt += self.batch_size * ep


            generated_images, self.last_pop, self.last_strat, fitnesses = main(self.disc_forward_pass,
                                                                     resolution=self.input_shape,
                                                                     wd=[self.run_dir,
                                                                           self.gan_images,
                                                                           self.best_im_dir,
                                                                           self.log_dir,
                                                                           self.images_to_evaluate],
                                                                     se=[epoch, step],
                                                                     population=self.last_pop,
                                                                     strat=self.last_strat) #chamada ao pylinhas
            self.last_gen_imgs = np.expand_dims(generated_images, axis=3)


            gen_output = self.discriminator(self.last_gen_imgs, training=True)
            real_output = self.discriminator(images, training=True)

            self.compute_losses(gen_output, real_output)

            gradients_of_discriminator = disc_tape.gradient(self.dloss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return fitnesses


    def train(self, epochs=1):
        start = time()

        for epoch in range(epochs):
            step = 0

            #generated_images
            for images in self.x_train:
                fitnesses = self.train_step(images, step, epoch)


                if self.log_losses: self.write_losses_epochs(step, epoch)

                self.generate_and_save_images(step + 1, epoch + 1, fitnesses)
                step += 1

                print('[DCGAN - step {}/{} of epoch {}/{}]:\t[Gloss, Dloss]: [{}, {}]\tTime so far: {} sec'.format(step,
                                                                                                                   len(self.x_train),
                                                                                                                   epoch + 1,
                                                                                                                   epochs,
                                                                                                                   self.gloss,
                                                                                                                   self.dloss,
                                                                                                                   time() - start))


        self.write_fid(n_epochs=epochs)

        self.training_time = time() - start
        self.plot_losses()  #"DESCOMENTEI" JESSICA
        return self.training_time, self.loss_hist

    def write_losses_epochs(self, step, epoch):
        fn = self.run_dir + "dcgan_losses.txt"
        with open(fn, mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            if epoch == 0 and step == 0:
                file.write("[d_loss, g_loss]\n")
            fwriter.writerow([self.dloss.numpy(), self.gloss.numpy()])


    def write_fid(self, n_epochs = 1):
        # get real images
        (x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        train_mask = np.isin(y_train, self.classes_to_train)
        x_train = x_train[train_mask]
        real_fid_batch = x_train[:self.batch_size * n_epochs]
        np.random.shuffle(real_fid_batch)

        # save files
        np.save(self.run_dir + real_fid_path_name, real_fid_batch, allow_pickle=False)
        np.save(self.run_dir + gen_fid_path_name, self.fid_gen_images, allow_pickle=False)


    def generate_and_save_images(self, s, e, fitnesses):

        self.last_gen_imgs = np.array(self.last_gen_imgs)
        self.last_gen_imgs = (self.last_gen_imgs + 1.0) * 127.5  # .... [-1, 1] to [0, 1]

        if s == len(self.x_train):
            if self.fid_gen_images is None:
                self.fid_gen_images = np.copy(self.last_gen_imgs)
            else:
                self.fid_gen_images = np.concatenate([self.fid_gen_images, self.last_gen_imgs], axis = 0)

        n_pop = self.last_gen_imgs.shape[0]
        n_col = 10 #forcei 10 colunas porque sim
        n_lin = round(n_pop/n_col)

        dloss_str = 'Discriminator Loss: {:05f}'.format(self.dloss)
        fig = plt.figure(figsize=(n_col,n_lin)).suptitle(dloss_str, fontsize = 10)

        max_fitness = -999
        max_fitness_index = 0

        for i in range(self.last_gen_imgs.shape[0]):
            fitness = fitnesses[i]
            plt.subplot(n_lin, n_col, i + 1).set_title('{:05f}'.format(fitness), y=-0.3 , fontsize = 7)

            im = self.last_gen_imgs[i, :, :].astype(np.uint8)
            if max_fitness < fitness:
                max_fitness = fitness
                max_fitness_index = i

            plt.imshow(im, cmap='gray')
            plt.axis('off')

        plt.savefig(self.gan_images + 'image_at_epoch{:04d}_step{:04d}.png'.format(e, s))
        plt.close()

        im_best = self.last_gen_imgs[max_fitness_index, :, :].astype(np.uint8)
        fig = plt.figure(frameon=False)
        dpi = 96
        fig.set_size_inches(self.gallery_res[0] / dpi, self.gallery_res[1] / dpi)
        plt.imshow(im_best, cmap ='gray')
        plt.axis('off')
        plt.savefig(self.best_for_generation + delimiter + "best_in_batch_epoch_{:04d}_step{:04d}.png".format(e, s), dpi=dpi)
        plt.close()


        #print("resolution2: ",self.input_shape)
        #save_image1(im_best, self.best_im_dir + delimiter + "best_in_batch_epoch_{:04d}_step{:04d}_teste.png".format(e, s), dims = self.input_shape)


    def plot_losses(self, show_graphics=False):
        fig, ax = plt.subplots(1, 1)
        ax.plot(range(len(self.loss_hist)), np.asarray(self.loss_hist)[:, 0], linestyle='-', label="D loss")
        pylab.legend(loc='upper left')
        ax.set_xlabel('Training steps')
        ax.set_ylabel('Loss')
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_title('Discriminator loss across training steps')
        fig.set_size_inches(12, 8)
        plt.savefig(fname=self.run_dir + 'Losses.svg', format="svg")
        if show_graphics: plt.show()
        plt.close(fig)


def set_experiment_ID_run():
    return str(int(time() * 1000.0) << 16)


if __name__ == '__main__':

    # Evolution Parameters
    gen_pop = POP_SIZE
    gens = [N_GENS]  #quantas geracoes quero treinar por cada training set
    epochs = EPOCHS
    runs = 1

    classes = [i for i in range(5, 10)]

    seeds = [random.randint(0, 0x7fffffff) for i in range(runs)]

    for r in range(runs):
        for c in classes:
            for g in gens:
                #for cur_set in fsets:
                print("doing: ", r, " class ", c, " for ", g, " generations, seed ", seeds[r])
                exp_prefix = "pylinhas_mnist_dynamicgens_" + str(c) + "_bin_fid"
                
                run_folder = exp_prefix + "_run_" + set_experiment_ID_run()
                mnist_dcgan = dcgan(batch_size=gen_pop, gens_per_batch=g, classes_to_train=c,
                						linear_gens_per_batch=True,
                                        run_folder = run_folder,
                                        seed=seeds[r],
                                        prefix=exp_prefix,
                                        log_losses=True,
                                        log_digits_class=False)
                train_time, train_hist = mnist_dcgan.train(epochs=epochs)
                print("Elapsed training time (s): ", train_time)
                    # mnist_dcgan.print_training_hist()
    print("Number of gen image: ", gen_image_cnt)
    print("Number of fake images: ", fake_image_cnt)