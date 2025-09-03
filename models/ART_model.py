import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import itertools
import pickle 
from util.image_pool import ImagePool

class ARTModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>

        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'NCE']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        visual_names_A_2 = ['real_B', 'fake_B_2', 'rec_A_2'] 
        visual_names_B_2 = ['real_B_virtual', 'fake_A_2', 'rec_B_2']

        self.i = 0
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            #self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']


        self.visual_names = visual_names_A + visual_names_B + visual_names_A_2 + visual_names_B_2  # combine visualizations for A and B
        

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'F', 'D_A', 'D_B', 'G_A_2', 'G_B_2', 'D_A_2', 'D_B_2']
        else:  # during test time, only load G
            self.model_names = ['G_A', 'G_B', 'G_A_2', 'G_B_2']

        # define networks (both generator and discriminator)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        self.netG_A_2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netG_B_2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)


        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            self.netD_A_2 = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD_B_2 = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_A_pool_2 = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool_2 = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()


            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netG_A_2.parameters(), self.netG_B_2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_A_2.parameters(), self.netD_B_2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G_2 = torch.optim.Adam(itertools.chain(self.netG_A_2.parameters(), self.netG_B_2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D_2 = torch.optim.Adam(itertools.chain(self.netD_A_2.parameters(), self.netD_B_2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # self.optimizers.append(self.optimizer_G_2)
            # self.optimizers.append(self.optimizer_D_2)


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.real_B_virtual = self.real_B_virtual[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.backward_D_A()
            self.backward_D_B() 
            self.backward_D_A_2() 
            self.backward_D_B_2()
            self.backward_G()
            # self.compute_D_loss().backward()                  # calculate gradients for D
            # self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_2, self.netD_B_2], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        #self.optimizer_G_2.zero_grad()
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        #if epoch > 30:
        #self.optimizer_G_2.step()

        # update G
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_2, self.netD_B_2], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        #self.optimizer_D_2.zero_grad()
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        #if epoch > 30:
        self.backward_D_A_2()      # calculate gradients for D_A
        self.backward_D_B_2()      # calculate graidents for D_B

        self.optimizer_D.step()  # update D_A and D_B's weights
        #self.optimizer_D_2.step()



        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_B_virtual = input['B_virtual'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        # if self.opt.flip_equivariance:
        #     self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
        #     if self.flipped_for_equivariance:
        #         self.real = torch.flip(self.real, [3])
        feat_q = self.netG_A(self.real_A, self.nce_layers, encode_only=True)
        #print('feat_q shape is ', feat_q[-1].shape)
        self.i += 1


        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        self.fake_B_2 = self.netG_A_2(self.fake_B)
        self.rec_A_2 = self.netG_B_2(self.fake_B_2) 
        self.fake_A_2 = self.netG_B_2(self.real_B_virtual) 
        self.rec_B_2 = self.netG_A_2(self.fake_A_2)




    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        #print('loss is ', loss_D)
        loss_D.backward()
        return loss_D




    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        #print('loss is ', self.loss_D_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)



    def backward_D_A_2(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B_2 = self.fake_B_pool_2.query(self.fake_B_2)
        self.loss_D_A_2 = self.backward_D_basic(self.netD_A_2, self.real_B_virtual, fake_B_2)

    def backward_D_B_2(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A_2 = self.fake_A_pool_2.query(self.fake_A_2)
        self.loss_D_B_2 = self.backward_D_basic(self.netD_B_2, self.real_B.detach(), fake_A_2)





    def backward_G(self, epoch=1):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = 0.5
        lambda_A = 10
        lambda_B = 10
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A_2 = self.netG_A_2(self.real_B_virtual)
            self.loss_idt_A_2 = self.criterionIdt(self.idt_A_2, self.real_B_virtual) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B_2 = self.netG_B_2(self.fake_B)
            self.loss_idt_B_2 = self.criterionIdt(self.idt_B_2, self.fake_B.detach()) * lambda_A * lambda_idt
        else:
            self.loss_idt_A_2 = 0
            self.loss_idt_B_2 = 0



        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        lambda_NCE = 1
        if lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
            self.loss_NCE_2 = self.calculate_NCE_loss_2(self.fake_B_2, self.real_A)
            #self.loss_NCE_cycle = self.calculate_NCE_loss(self.real_B, self.fake_A) 
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0



        #if epoch > 30:
        # GAN loss D_A(G_A(A))
        self.loss_G_A_2 = self.criterionGAN(self.netD_A_2(self.fake_B_2), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B_2 = self.criterionGAN(self.netD_B_2(self.fake_A_2), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A_2 = self.criterionCycle(self.rec_A_2, self.fake_B) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B_2 = self.criterionCycle(self.rec_B_2, self.real_B_virtual) * lambda_B


        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + lambda_NCE + self.loss_NCE + self.loss_NCE_2
        self.loss_G += self.loss_G_A_2 + self.loss_G_B_2 + self.loss_cycle_A_2 + self.loss_cycle_B_2 + self.loss_idt_A_2 + self.loss_idt_B_2
        
        #print('!loss_G_1 is ', self.loss_G)
        self.loss_G.backward()

        #if epoch > 30:
        # combined loss and calculate gradients
        self.loss_G_2 = self.loss_G_A_2 + self.loss_G_B_2 + self.loss_cycle_A_2 + self.loss_cycle_B_2 + self.loss_idt_A_2 + self.loss_idt_B_2
        #print('?loss_G_2 is ', self.loss_G_2)
        #self.loss_G_2.backward()









    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)
        #print('feat q shape is ', feat_q[-1].shape)
        # if self.opt.flip_equivariance and self.flipped_for_equivariance:
        #     feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG_A(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers



    def calculate_NCE_loss_2(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)

        # if self.opt.flip_equivariance and self.flipped_for_equivariance:
        #     feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG_B_2(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
