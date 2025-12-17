# SNRAware

This repository contains the Pytorch code in our paper [SNRAware: Improved Deep Learning MRI Denoising with Signal-to-noise Ratio Unit Training and G-factor Map Augmentation](https://pubs.rsna.org/doi/full/10.1148/ryai.250227) published at the Radiology: Artificial Intelligence:

```latex
@article{
    doi:10.1148/ryai.250227,
    author = {Xue, Hui and Hooper, Sarah M. and Pierce, Iain and Davies, Rhodri H. and Stairs, John and Naegele, Joseph and Campbell-Washburn, Adrienne E. and Manisty, Charlotte and Moon, James C. and Treibel, Thomas A. and Hansen, Michael S. and Kellman, Peter},
    title = {SNRAware: Improved Deep Learning MRI Denoising with Signal-to-noise Ratio Unit Training and G-factor Map Augmentation},
    journal = {Radiology: Artificial Intelligence},
    volume = {7},
    number = {6},
    pages = {e250227},
    year = {2025},
    doi = {10.1148/ryai.250227},
    note ={PMID: 41123451},
    URL = {https://doi.org/10.1148/ryai.250227}
}
```

- Model type: Imaging AI, non-generative
- License: MIT

## Get started

[just](https://github.com/casey/just), [uv]() and [gif-lfs]() are used in this project. If not, please install this tool:

```bash
# install just
curl -fsSL https://just.systems/install.sh | sudo bash -s -- --to /usr/local/bin

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install git-lfs
sudo apt update
sudo apt install git-lfs
```

Make sure commands `just`, `us` are on your path.

Also, this project requires NVIDIA GPU. To check whether your GPU is available and is working:

```bash
nvidia-smi
```

Then, please set up the virtual environment and run tests:

```bash
# set up env
uv sync

# pull down test data
git lfs pull

# run the test
uv run pytest -m gpu ./test
```

## Data
Dataset for MR denoising training is not opened at this moment.

## Model
Three models are released at https://huggingface.co/microsoft/SNRAware

- SNRAware-small: 27.7million parameters
- SNRAware-medium: 55.1million parameters
- SNRAware-large: 109million parameters

## Direct intended uses
SNRAware is shared for research and technical development purposes only, to denoise MR images.

## License and Usage Notices
The data, code, and model checkpoints described in this repository is provided for research and technical development use
only. The data, code, and model checkpoints are not intended for use in clinical use.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Documentation

Please find documentation in the [docs/overview](./docs/overview.md).
