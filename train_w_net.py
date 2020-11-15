import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import kornia

# import net_depth.burst_net as burst_net
import enhance_net_3 as enh
from net_depth.w_net_1 import BurstNet
from net_depth.data_load_proc import Dataset_Full
# import enhance_net
import utils
import numpy as np
import plenopticIO.imgIO as rtxIO

import time

Axes3D = Axes3D


def main():

    # torch.autograd.set_detect_anomaly(True)
    # Parameters
    batch_size = 8
    num_epochs = 10
    num_workers = 4
    shuffle_data = True
    lr = 0.0001

    load_best = False

    patch_locs = '/home/carson/hq_raytrix_images/patch_lens_0.01/train/'

    model_save_name = '/home/carson/libs/pytorch_models/w_net_2/w_net_2_001_{}_l_{:.3}.pt'
    best_loc = '/home/carson/libs/pytorch_models/w_net_2/patch_net_1_001_2_l_0.000766.pt'

    if load_best:
        checkpoint = torch.load(best_loc)

    # Setup tensorboard
    comment = " w_net_2_001 lr {} batch {}".format(lr, batch_size)

    params = {'batch_size': batch_size,
              'shuffle': shuffle_data,
              'num_workers': num_workers}

    # Set an initial loss value
    previous_loss = 1000000.

    tb = SummaryWriter(comment=comment)

    patch_loc_list = utils.data_lists(patch_locs)

    # Setup data generators
    training_set = Dataset_Full(patch_loc_list)
    training_gen = data.DataLoader(training_set, **params)

    # Check cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('use_cuda: {}'.format(use_cuda))

    # Load depth net
    print('Loading burst net')
    time.sleep(1)
    u_1 = enh.UNet(num_channels=18)
    u_2 = enh.UNet(num_channels=6)

    u_1 = torch.nn.DataParallel(u_1, device_ids=[device])
    u_2 = torch.nn.DataParallel(u_2, device_ids=[device])

    # is_true = True
    if True:
        print('Loading best u_1')
        net_loc = '/home/carson/libs/pytorch_models/w_net_1/w_net_1_001_first_u_9_l_0.00176.pt'
        checkpoint = torch.load(net_loc)
        new_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            k_s = k.split('.')
            del k_s[1]
            new_k = '.'.join(k_s)
            new_state_dict[new_k] = v
        u_1.load_state_dict(new_state_dict)

    burst_net = BurstNet(u_1, u_2, freeze_u1=False)

    # Set criterion and optimizer
    criterion_1 = nn.MSELoss()
    criterion_2 = nn.MSELoss()

    # Get number of trainable parameters
    trainable_params = sum(p.numel() for p in burst_net.parameters() if p.requires_grad)

    print('Trainable parameters: {}'.format(trainable_params))

    print('Optimizer seeing all parameters!')
    optimizer = torch.optim.Adam(burst_net.parameters(), lr=lr)
    if load_best:
        optimizer.load_state_dict(checkpoint['opt_state_dict'])

    print('Entering training loop')
    num_samples = len(training_gen)
    num_samples_actual = num_samples * 0.25

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 5], gamma=0.1)

    if load_best:
        epoch = checkpoint['epoch'] + 1
    else:
        epoch = 0

    while epoch < num_epochs:
        avg_loss = 0
        loss_count = 0
        stp_avg_loss = 0
        stp_loss_count = 0
        # if epoch == 3:
        #     print('Unfreezing u_1')
        #     for name, child, in burst_net.u_1.named_children():
        #         for c_name, c_param in child.named_parameters():
        #             c_param.requires_grad = True
        #
        #     # Get number of trainable parameters
        #     trainable_params = sum(p.numel() for p in burst_net.parameters() if p.requires_grad)
        #
        #     print('Trainable parameters: {}'.format(trainable_params))


        for i_val, img_data in enumerate(training_gen):
            # if i_val > 40:
            #     break
            # print(i_val)

            inp_patch, center_patch, ref_patch = img_data

            inp_patch, center_patch, ref_patch = inp_patch.to(device), center_patch.to(device), ref_patch.to(device)

            # Zero gradients
            burst_net.zero_grad()
            optimizer.zero_grad()

            # Upsample lenses
            net_out, u_out_1 = burst_net(inp_patch, center_patch)
            # net_out, u_out_1 = burst_net(center_patch)

            # Calculate loss
            # first u loss
            loss_1 = criterion_1(u_out_1, ref_patch)
            # second u loss
            loss_2 = criterion_2(net_out, ref_patch)

            avg_loss += loss_2.item()
            loss_count += 1
            stp_avg_loss += loss_2.item()
            stp_loss_count += 1

            # Perform backward pass
            loss_tot = loss_1 + loss_2
            loss_tot.backward()

            # zero and step optimizer
            optimizer.step()

            # # Update tb
            if i_val % 100 == 0:
                tb.add_scalar("loss", stp_avg_loss / stp_loss_count, (epoch * num_samples + i_val))
                plt.close()
                prog_fig = plt.figure()

                img_1 = center_patch[0].cpu().detach().permute(1, 2, 0)[:, :, :3]
                img_2 = net_out[0].cpu().detach().permute(1, 2, 0)
                img_3 = ref_patch[0].cpu().detach().permute(1, 2, 0)
                img_4 = u_out_1[0].cpu().detach().permute(1, 2, 0)

                # Get rid of the matplotlib error.
                img_1 = np.where(img_1 > 1, 1, img_1)
                img_1 = np.where(img_1 < 0, 0, img_1)
                img_2 = np.where(img_2 > 1, 1, img_2)
                img_2 = np.where(img_2 < 0, 0, img_2)
                img_3 = np.where(img_3 > 1, 1, img_3)
                img_3 = np.where(img_3 < 0, 0, img_3)
                img_4 = np.where(img_4 > 1, 1, img_4)
                img_4 = np.where(img_4 < 0, 0, img_4)

                prog_fig.add_subplot(141)
                plt.imshow(img_4)
                prog_fig.add_subplot(142)
                plt.imshow(img_1)
                prog_fig.add_subplot(143)
                plt.imshow(img_2)
                prog_fig.add_subplot(144)
                plt.imshow(img_3)

                tb.add_figure('prog', prog_fig, global_step=epoch * num_samples + i_val)

                stp_avg_loss = 0
                stp_loss_count = 0

        if avg_loss/loss_count < previous_loss:
            print("Epoch: {}, Average Loss improved from {} to {}".format(epoch, previous_loss, avg_loss / loss_count))
            previous_loss = avg_loss / loss_count

            model_file_name = model_save_name.format(epoch, avg_loss / loss_count)

            torch.save({'model_state_dict': burst_net.state_dict(),
                        'epoch': epoch,
                        'opt_state_dict': optimizer.state_dict()}, model_file_name)

        else:
            print("Epoch: {}, Loss {}".format(epoch, avg_loss / loss_count))

        epoch += 1

        # utils.validate_on_test_patch(patch_net=burst_net)

        scheduler.step()

    tb.close()


if __name__ == "__main__":
    main()
