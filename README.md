# Greedy-DiM <br><sub>The official implementation of the IJCB 2024 paper</sub>

![](https://img.shields.io/badge/last%20update-2024.09.24-green.svg)
![](https://img.shields.io/badge/python-3.8-green.svg)
![](https://img.shields.io/badge/pytorch-2.0.1-green.svg)
![](https://img.shields.io/badge/cudnn-8.5.0-green.svg)
![](https://img.shields.io/badge/license-TODO-green.svg)

![Teaser image](./docs/assets/greedy_dim_morph_comp.png)

**Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs**<br>
Zander W. Blasingame and Chen Liu<br>
[arXiv](https://arxiv.org/abs/2404.06025)

Abstract: *Morphing attacks are an emerging threat to state-of-the-art Face Recognition (FR) systems, which aim to create a single image that contains the biometric information of multiple identities. Diffusion Morphs (DiM) are a recently proposed morphing attack that has achieved state-of-the-art performance for representation-based morphing attacks. However, none of the existing research on DiMs have leveraged the iterative nature of DiMs and left the DiM model as a black box, treating it no differently than one would a Generative Adversarial Network (GAN) or Variational AutoEncoder (VAE). We propose a greedy strategy on the iterative sampling process of DiM models which searches for an optimal step guided by an identity-based heuristic function. We compare our proposed algorithm against ten other state-of-the-art morphing algorithms using the open-source SYN-MAD 2022 competition dataset. We find that our proposed algorithm is unreasonably effective, fooling all of the tested FR systems with an MMPMR of 100%, outperforming all other morphing algorithms compared.*

## Installation
Sign the [code request form](CITeR_SoftwareReleaseAgreeement.docx) and send it to [citer@clarkson.edu](mailto:citer@clarkson.edu?subject=[GitHub]%20DiM%20Source%20Code%20Request)

Run the script `installer.sh`. This script will setup the virtualenv and install all dependencies for the project.

Code is named `greedy_dim.py` in the `diffae` directory, directory is installed automatically by the install script
Run `source venv/bin/activate` to activate the virtual environment
Program should install all needed python packages for the morphing code (verified on 2024.09.24)

### Setting up the U-Net
Install the FFHQ 256 checkpoint from the [diffae repository](https://github.com/phizaz/diffae) and place in the `checkpoints` directory

### Setting up the Identity Loss
The primary identity loss makes use of ArcFace. To download the same version we used in our experiments navigate to the [arcface repository](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) and download the `glint360k_cosface_r100_fp16_0.1` folder and place it in the `backbones` folder

## Usage
Program takes a CSV consisting of rows in the following format
```csv
path/to/source/id_a.png,path/to/source/id_b.png,path/to/output/morph_a_b.png
path/to/source/id_c.png,path/to/source/id_d.png,path/to/output/morph_c_d.png
```

Run 'python make_morphs.py -h' to see usage details


## Citation
If you found this codebase useful in your research, please consider citing either or both of:
```bibtex
@INPROCEEDINGS{blasingame_greedy_dim,
      title={Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs},
      booktitle={2024 IEEE International Joint Conference on Biometrics (IJCB)}, 
      author={Blasingame, Zander W. and Liu, Chen},
      year={2024},
      pages={1-10},
      url={https://arxiv.org/abs/2404.06025}, 
}

@article{blasingame_dim,
   title={Leveraging Diffusion for Strong and High Quality Face Morphing Attacks},
   volume={6},
   ISSN={2637-6407},
   url={http://dx.doi.org/10.1109/TBIOM.2024.3349857},
   DOI={10.1109/tbiom.2024.3349857},
   number={1},
   journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Blasingame, Zander W. and Liu, Chen},
   year={2024},
   month=jan, pages={118–131}}
```
