# Cog depth-anything-video-SBS Model

This is a custom implementation of [JohanDL/Depth-Anything-Video](https://huggingface.co/spaces/JohanDL/Depth-Anything-Video) as a [Cog](https://github.com/replicate/cog) model to produce a stereo video output.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

Run a prediction:

    cog predict -i video=@wooly.mp4

## Output

![output](output.gif)

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

`depth_anything/` comes from [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything), which is Apache 2.0.

The model weights are not covered by this license. [LiheYoung/depth-anything-small-hf](https://huggingface.co/LiheYoung/depth-anything-small-hf) is released under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
