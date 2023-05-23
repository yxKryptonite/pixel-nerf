# pixelNeRF for Computational Photography, Spring 2023

*Yuxuan Kuang* and *Shaofan Sun*

[Original README](./README_original.md)

**Warning: this is not an official product but only a confirmatory repo to realize our ideas. For reproducing our work, you should follow the environment setup of pixelNeRF and install [CLIP](https://github.com/openai/CLIP).**

## Changes

- Add `eval/eval_camera.py` to test the correspondence of image quality with camera poses.
- Reconstruct `eval/eval_real.py` to better evaluate images on real scenes. (Turn original script into `eval/eval_real_original.py`).
- Take our own photos at `my_input/` and test them with `eval/eval_real.py`, output is at `my_output/`.

## Improvement

- Add `src/camera/` to add some search utils and loss functions, including using VGG-16 to measure visual loss and using CLIP to measure image feature similarities.
- Figure out that `radius` is crucial to the image quality and implement a searching algorithm for a good `radius` to improve the render quality.
- Re-implement (actually debug for original code) the reference code for single-view 3D reconstruction, fast and reliable!

## How to Run

```shell
python eval/eval_real.py -n sn64 -I my_input -O my_output --gpu_id 5   
```

## Division of Labor

*Yuxuan Kuang*: code implementation, experiment design, report writing

*Shaofan Sun*: data collection, report writing