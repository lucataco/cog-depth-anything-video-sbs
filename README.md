# Cog depth-anything-video-SBS Model

This is a custom implementation of [JohanDL/Depth-Anything-Video](https://huggingface.co/spaces/JohanDL/Depth-Anything-Video) as a [Cog](https://github.com/replicate/cog) model to produce a stereo video output.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

Run a prediction:

    cog predict -i video=@wooly.mp4

## Output

![output](output.gif)