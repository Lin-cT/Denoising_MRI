# SNRAware

This repository contains the Pytorch code in our paper [SNRAware: Improved Deep Learning MRI Denoising with Signal-to-noise Ratio Unit Training and G-factor Map Augmentation](https://pubs.rsna.org/doi/full/10.1148/ryai.250227) published at the Radiology: Artificial Intelligence:

```latex
@article{doi:10.1148/ryai.250227,
author = {Xue, Hui and Hooper, Sarah M. and Pierce, Iain and Davies, Rhodri H. and Stairs, John and Naegele, Joseph and Campbell-Washburn, Adrienne E. and Manisty, Charlotte and Moon, James C. and Treibel, Thomas A. and Hansen, Michael S. and Kellman, Peter},
title = {SNRAware: Improved Deep Learning MRI Denoising with Signal-to-noise Ratio Unit Training and G-factor Map Augmentation},
journal = {Radiology: Artificial Intelligence},
volume = {0},
number = {ja},
pages = {e250227},
year = {0},
doi = {10.1148/ryai.250227},
    note ={PMID: 41123451},
URL = {    
        https://doi.org/10.1148/ryai.250227
},
eprint = {   
        https://doi.org/10.1148/ryai.250227
},
abstract = { Purpose To develop and evaluate a novel deep learning-based MRI denoising method using quantitative noise distribution information obtained during image reconstruction to improve model performance and generalization. Materials and Methods This retrospective study included a training set of 2885236 images from 96605 cardiac cine series acquired on 3T MRI scanners from January 2018 to December 2020. 95\% of these data were used for training and 5\% for validation. The hold-out test set included 3000 cine series, acquired in the same period. Fourteen model architectures were evaluated by instantiating each of the two backbone types with seven transformer and convolution block types. The proposed SNRAware training scheme leveraged MRI reconstruction knowledge to enhance denoising by simulating diverse synthetic datasets and providing quantitative noise distribution information. Internal testing measured performance using peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM), whereas external tests conducted on 1.5T real-time cardiac cine, first-pass cardiac perfusion, brain, and spine MRIs assessed generalization across various sequences, contrasts, anatomies, and field strengths. Results SNRAware improved performance on internal tests conducted on a hold-out dataset of 3000 cine series. Models trained without reconstruction knowledge achieved the worst performance metrics. Improvement was architecture-agnostic for both convolution and transformer models; however, transformer models outperformed their convolutional counterparts. Additionally, 3D input tensors showed improved performance over 2D images. The best-performing model from the internal testing generalized well to external samples, delivering 6.5 × and 2.9 × contrast-to-noise ratio improvement for real-time cine and perfusion imaging, respectively. The model trained using only cardiac cine data generalized well to T1 MPRAGE (Magnetization-Prepared Rapid Gradient-Echo) brain 3D and T2 TSE (turbo spin-echo) spine MRIs. Conclusion The SNRAware training scheme leveraged data obtained during the image reconstruction process for deep learning-based MRI denoising training, resulting in improved performance and good generalization. © The Author(s) 2025. Published by the Radiological Society of North America under a CC BY 4.0 license. }
}
```

- Model type: Imaging AI, non-generative
- License: MIT

## Get started

[just](https://github.com/casey/just) is used in this project. If not, please install this tool:

```bash
# install just
wget -qO - 'https://proget.makedeb.org/debian-feeds/prebuilt-mpr.pub' | gpg --dearmor | sudo tee /usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg 1> /dev/null
echo "deb [arch=all,$(dpkg --print-architecture) signed-by=/usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg] https://proget.makedeb.org prebuilt-mpr $(lsb_release -cs)" | sudo tee /etc/apt/sources.list.d/prebuilt-mpr.list
sudo apt update
sudo apt install just -y
```

Then, please set up the virtual environment and run tests:

```bash
# show the list
just --list

# set up virtual environment
just setup-env

# review documentation
just serve-docs

# run test
just test
```

## Data
Dataset for MR denoising training is not opened at this moment.

## Run model training

## Direct intended uses
SNRAware is shared for research purposes only, namely, benchmarking and inference on downstream
tasks. This is a research model which should not be used in any clinical or production scenario.

## License and Usage Notices
The data, code, and model checkpoints described in this repository is provided for research use
only. The data, code, and model checkpoints is not intended for use in clinical use.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.