import torch
from options import TrainOptions
from datasets import dataset_multi
from model import MD_multi
from saver import save_imgs, save_concat_imgs, Saver
import os
import numpy as np

def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    # data loader
    print('\n--- load dataset ---')
    dataset = dataset_multi(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=False,
                                               num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = MD_multi(opts)
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)
    model.eval()

    # saver for display and output
    saver = Saver(opts)

    # test
    print('\n--- testing ---')
    num_max_sample = 1
    ep = 0
    for idx in range(num_max_sample):
        for i, (images, c_org) in enumerate(train_loader):
            if images.size(0) != opts.batch_size:
                continue
            print('{}/{}'.format(i, len(train_loader)))

            # input data
            images = images.cuda(opts.gpu).detach()
            c_org = c_org.cuda(opts.gpu).detach()

            # do not update model
            with torch.no_grad():
                if opts.isDcontent:
                    if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                        model.update_D_content(images, c_org)    # wait to modify
                        continue
                    else:
                        # only forward, no backward
                        model.no_update_D(images, c_org)
                        model.update_EG()
                else:
                    # only forward, no backward
                    model.no_update_D(images, c_org)
                    #model.update_EG()

                # save result image
                saver.write_img(ep, model)
                ep += 1
    return


if __name__ == '__main__':
    main()
