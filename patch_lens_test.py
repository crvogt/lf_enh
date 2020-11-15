# Testing the new w_net format
# First u handles warping the external lenses
# second u combines the two
import plenopticIO.imgIO as rtxIO
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import utils
import enhance_net_3 as enh
import rendering.render as rtxrend

def center_crop_ends(pc, x_diff, y_diff, lf_shape):
    # Getting distances from center lens, will determine all patch sizes
    # Notation based on sketch in notebook
    lx = pc[1]
    a = lx - x_diff

    j = lf_shape[1]
    b = j - lx - x_diff

    ly = pc[0]
    f = ly - y_diff

    k = lf_shape[0]
    e = k - ly - y_diff

    return a, b, e, f

def crop_ends(pc, a, b, e, f):
    xs = pc[1] - a
    xe = pc[1] + b

    ys = pc[0] - f
    ye = pc[0] + e

    return xs, xe, ys, ye

def main():
    with torch.no_grad():
        # LF of interet
        exp_val = 55

        lf_loc = '/home/carson/hq_raytrix_images/data/20/proc/0.5.png'
        ref_loc = '/home/carson/hq_raytrix_images/data/20/proc/10.png'
        u2_loc = '/home/carson/libs/pytorch_models/final_models/w_net_2.pt'

        ll_lf = plt.imread(lf_loc)[:, :, :3]
        lf_shape = ll_lf.shape

        # As per the training...
        sf = 5
        pcoord_dict = rtxIO.load_scene_pcoord(sf=sf)

        x_diff = np.abs(pcoord_dict[(0, 0)][1] - pcoord_dict[(0, 1)][1]) + 3
        y_diff = np.abs(pcoord_dict[(0, 0)][0] - pcoord_dict[(1, -1)][0]) + 19

        if sf > 1:
            ll_lf = cv2.resize(ll_lf, (lf_shape[0] * sf, lf_shape[1] * sf))
            lf_shape = ll_lf.shape

        a, b, e, f = center_crop_ends(pcoord_dict[0, 0], x_diff, y_diff, lf_shape)

        # Now find the lenses
        neighbors = [(0, 0), (0, 1), (1, 0), (1, -1), (0, -1), (-1, 1), (-1, 0)]
        center_patch = 0
        inp_patch = 0
        patch_shape = None
        for n_val, neighbor in enumerate(neighbors):
            if n_val == 0:
                xs, xe, ys, ye = crop_ends(pcoord_dict[neighbor], a, b, e, f)
                center_patch = ll_lf[ys:ye, xs:xe, :]
                patch_shape = list(center_patch.shape)

                patch_shape[0] = int(round(patch_shape[0] / sf))
                patch_shape[1] = int(round(patch_shape[1] / sf))

                center_patch = cv2.resize(center_patch, (patch_shape[1], patch_shape[0]))

            elif n_val == 1:
                xs, xe, ys, ye = crop_ends(pcoord_dict[neighbor], a, b, e, f)
                inp_patch = ll_lf[ys:ye, xs:xe, :]
                inp_patch = cv2.resize(inp_patch, (patch_shape[1], patch_shape[0]))
            else:
                xs, xe, ys, ye = crop_ends(pcoord_dict[neighbor], a, b, e, f)
                tmp_patch = ll_lf[ys:ye, xs:xe, :]
                tmp_patch = cv2.resize(tmp_patch, (patch_shape[1], patch_shape[0]))
                inp_patch = np.concatenate((inp_patch, tmp_patch), axis=2)

        to_tensor = torchvision.transforms.ToTensor()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        exposure_lim = 1000
        multi_val = exposure_lim / exp_val

        inp_u1 = to_tensor(inp_patch * multi_val)
        inp_u1 = inp_u1.view(1, *inp_u1.shape).cuda()

        u_1 = enh.UNet(num_channels=18)
        u_1 = torch.nn.DataParallel(u_1, device_ids=[device])

        state_dict = torch.load(u2_loc)['model_state_dict']
        u1_state_dict = dict()
        u2_state_dict = dict()
        for k, v in state_dict.items():
            k_s = k.split('.')
            if k_s[0] == 'u_1':
                del k_s[0]
                new_k = '.'.join(k_s)
                u1_state_dict[new_k] = v
            elif k_s[0] == 'u_2':
                del k_s[0]
                new_k = '.'.join(k_s)
                u2_state_dict[new_k] = v

        u_1.load_state_dict(u1_state_dict)

        u_2 = enh.UNet(num_channels=6)
        u_2 = torch.nn.DataParallel(u_2, device_ids=[device])
        u_2.load_state_dict(u2_state_dict)
        u_2.eval()

        center_patch *= multi_val
        center_patch = to_tensor(center_patch)
        center_patch = center_patch.view(1, *center_patch.shape).cuda()
        print('into u1')
        u_1_out = u_1(inp_u1)

        print('u1_complete')

        del u_1
        del inp_u1
        del u1_state_dict
        del state_dict
        torch.cuda.empty_cache()

        inp_u2 = torch.cat((center_patch, u_1_out), dim=1)
        out_img_u_2 = u_2(inp_u2).cpu().detach().squeeze().permute(1, 2, 0).numpy()

        print('u2 complete')
        # plt.imshow(out_img_u_2)
        # plt.show(block=False)
        # plt.pause(5)

        out_shape = out_img_u_2.shape
        out_img_u_2 = cv2.resize(out_img_u_2, (out_shape[0] * sf, out_shape[1] * sf))
        plt.imshow(out_img_u_2)
        plt.show()
        
        out_lenses = rtxIO.load_scene_dl_img_basic(out_img_u_2, sf=sf)

        lenses = rtxIO.load_scene_pcoord(sf=1)

        utils.remove_neighbors(lenses)
        # utils.remove_neighbors(lenses)
        # utils.remove_neighbors(out_lenses)

        print(len(out_lenses), len(lenses))

        lens_shape = 24
        lens_keys = list(out_lenses.keys())
        save_len = len(lens_keys)

        enh_lenses_img = np.zeros((lens_shape, lens_shape * save_len, 3))
        for l_val, lcoord in enumerate(lens_keys):
            sa = l_val * lens_shape
            se = l_val * lens_shape + lens_shape
            enh_lenses_img[:, sa:se, :] = cv2.resize(out_lenses[lcoord], (24, 24))

        # plt.imsave('patch_lens_depth.png', enh_lenses_img)

        # rendered_img = rtxrend.render_lens_imgs(lenses, out_lenses)
        #
        # # now do the same for ref
        # ref = cv2.resize(plt.imread(ref_loc)[:, :, :3], (2048*5, 2048*5))
        #
        # ref_lens_imgs = rtxIO.load_scene_dl_img_basic(ref, sf=5, pad_val=5)
        # for lc in ref_lens_imgs:
        #     ref_lens_imgs[lc] = cv2.resize(ref_lens_imgs[lc], (24, 24))
        # utils.remove_neighbors(ref_lens_imgs)
        # utils.remove_neighbors(ref_lens_imgs)
        #
        # print(len(ref_lens_imgs))
        #
        # ref_rendered = rtxrend.render_lens_imgs(lenses, ref_lens_imgs)
        #
        # import skimage
        #
        # psnr = skimage.measure.compare_psnr(ref_rendered, rendered_img)
        # ssim = skimage.measure.compare_ssim(ref_rendered, rendered_img, multichannel=True)
        # print(psnr, ssim)
        #
        # fig_ = plt.figure()
        # # fig_.add_subplot(121)
        # plt.imshow(rendered_img)
        # # fig_.add_subplot(122)
        # # plt.imshow(ref_rendered)
        # plt.show(block=False)
        # plt.pause(8)

if __name__ == '__main__':
    main()
