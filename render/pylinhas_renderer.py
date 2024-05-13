import cairo
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from render.renderinterface import RenderingInterface
from utils import map_number

class PylinhasRenderer(RenderingInterface):
    def __init__(self, args):
        super(PylinhasRenderer, self).__init__(args)

        self.img_size = args.img_size
        self.num_lines = args.num_lines

        self.device = args.device

        self.header_length = 1

        # genes per line
        self.genotype_size = 5 #x1, y1, x2, y2, stroke (antes era 8 porque tinha as cores tambem rgb)

        self.individual = None

    def chunks(self, array):
        return np.reshape(array, (self.num_lines, self.genotype_size))

    def generate_individual(self):
        return np.random.rand(self.num_lines, self.genotype_size).flatten()

    def to_adam(self, individual, gradients=True):
        self.individual = np.copy(individual)
        self.individual = self.chunks(self.individual)
        self.individual = torch.tensor(self.individual).float().to(self.device)

        if gradients:
            self.individual.requires_grad = True

        optimizer = torch.optim.Adam([self.individual], lr=1.0)

        return [optimizer]

    def get_individual(self):
        return self.individual.cpu().detach().numpy().flatten()

    def __str__(self):
        return "pylinhas"

    def render(self, image_size=28, filename = ""):
        input_ind = self.individual
        input_ind = input_ind.cpu().detach().numpy()

        # split input array into header and rest
        head = input_ind[:self.header_length]
        rest = input_ind[self.header_length:]

        # determine background color from header
        R = 0#head[0][0]
        G = 0#head[0][1]
        B = 0#head[0][2]

        # create the image and drawing context
        ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, image_size, image_size)
        cr = cairo.Context(ims)

        cr.set_source_rgba(R, G, B, 1.0)  # everythingon cairo appears to be between 0 and 1
        cr.rectangle(0, 0, image_size, image_size)  # define a recatangle and then fill it
        cr.fill()

        # now draw lines
        if len(head[0]) > 8:
            min_width = 0.004 * image_size
            max_width = 0.04 * image_size
        else:
            min_width = 0.0001 * image_size
            max_width = 0.1 * image_size

        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)

        for e in rest:
            # determine foreground color from header
            R = 1
            G = 1
            B = 1
            w = map_number(e[0], 0, 1, min_width, max_width)

            cr.set_source_rgb(R, G, B)
            cr.set_line_width(w)    # line width

            for it in range(1, len(e) - 1, 2):#for it in range(4, len(e) - 1, 2):
                x = map_number(e[it], 0, 1, 0, image_size)
                y = map_number(e[it + 1], 0, 1, 0, image_size)
                cr.line_to(x, y)

            cr.close_path()
            cr.stroke_preserve()
            cr.fill()

        pilMode = 'RGB'
        argbArray = np.fromstring(bytes(ims.get_data()), 'c').reshape(-1, 4)
        rgbArray = argbArray[:, 2::-1]
        pilData = rgbArray.reshape(-1).tostring()
        pilImage = Image.frombuffer(pilMode,
                                    (ims.get_width(), ims.get_height()), pilData, "raw",
                                    pilMode, 0, 1)
        pilImage = pilImage.convert('RGB')

        return TF.to_tensor(pilImage).unsqueeze(0)