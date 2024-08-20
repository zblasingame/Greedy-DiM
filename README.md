# Greedy-DiM <br><sub>The official implementation of the IJCB 2024 paper</sub>

**Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs**<br>
Zander W. Blasingame and Chen Liu<br>
[arXiv](https://arxiv.org/abs/2404.06025)

Abstract: *Morphing attacks are an emerging threat to state-of-the-art Face Recognition (FR) systems, which aim to create a single image that contains the biometric information of multiple identities. Diffusion Morphs (DiM) are a recently proposed morphing attack that has achieved state-of-the-art performance for representation-based morphing attacks. However, none of the existing research on DiMs have leveraged the iterative nature of DiMs and left the DiM model as a black box, treating it no differently than one would a Generative Adversarial Network (GAN) or Varational AutoEncoder (VAE). We propose a greedy strategy on the iterative sampling process of DiM models which searches for an optimal step guided by an identity-based heuristic function. We compare our proposed algorithm against ten other state-of-the-art morphing algorithms using the open-source SYN-MAD 2022 competition dataset. We find that our proposed algorithm is unreasonably effective, fooling all of the tested FR systems with an MMPMR of 100%, outperforming all other morphing algorithms compared.*

 ## Citation
```bibtex
@misc{blasingame_greedy_dim,
      title={Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs}, 
      author={Zander W. Blasingame and Chen Liu},
      year={2024},
      eprint={2404.06025},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.06025}, 
}
```
