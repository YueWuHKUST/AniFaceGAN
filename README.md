# AniFaceGAN: Animatable 3D-Aware Face Image Generation for Video Avatars

This is a pytorch implementation of the following paper:

Yue Wu, Yu Deng, Jiaolong Yang, Fangyun Wei, Qifeng Chen, Xin Tong.  **AniFaceGAN: Animatable 3D-Aware Face Image Generation
for Video Avatars**, NeurIPS 2022 (Spotlight).

### [Project page](https://yuewuhkust.github.io/AniFaceGAN/) | [Paper](https://arxiv.org/abs/2210.06465)

Abstract: _Although 2D generative models have made great progress in face image generation and animation, they often suffer from undesirable artifacts such as 3D inconsistency when rendering images from different camera viewpoints. This prevents them from synthesizing video animations indistinguishable from real ones. Recently, 3D-aware GANs extend 2D GANs for explicit disentanglement of camera pose by leveraging 3D scene representations. These methods can well preserve the 3D consistency of the generated images across different views, yet they cannot achieve fine-grained control over other attributes, among which facial expression control is arguably the most useful and desirable for face animation. In this paper, we propose an animatable 3D-aware GAN for multiview consistent face animation generation. The key idea is to decompose the 3D representation of the 3D-aware GAN into a template field and a deformation field, where the former represents different identities with a canonical expression, and the latter characterizes expression variations of each identity. To achieve meaningful control over facial expressions via deformation, we propose a 3D-level imitative learning scheme between the generator and a parametric 3D face model during adversarial training of the 3D-aware GAN. This helps our method achieve high-quality animatable face image generation with strong visual 3D consistency, even though trained with only unstructured 2D images. Extensive experiments demonstrate our superior performance over prior works.._

## Enviroment
The code is tested in the docker environment: yuewuust/pytorch1.11.0_nviffrast:v11

Please refer to [Link](https://hub.docker.com/r/yuewuust/pytorch1.11.0_nviffrast/tags).

## Test
The expression coefficients are extracted by [Deep3DFaceRecon](https://github.com/microsoft/Deep3DFaceReconstruction). And we provide the smile expression ./mat/01626.mat as an example. Zero expression is defined as a neutral face. 

Run the 
```
./render.sh
``` 
and the results will be sorted in ./multiview_imgs/.

## To do

- [X] Release inference code
- [X] Release pretrained checkpoints
- [ ] Clean up code.
- [ ] Add detailed instrunctions.


## Citation

Please cite the following paper if this work helps your research:

    @inproceedings{yue2022anifacegan,
    title={AniFaceGAN: Animatable 3D-Aware Face Image Generation for Video Avatars},
    author={Wu, Yue and Deng, Yu and Yang, Jiaolong and Wei, Fangyun and Chen Qifeng and Tong, Xin},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}
