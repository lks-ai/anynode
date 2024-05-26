# AnyNode v0.1 (🍄 beta)

A ComfyUI Node that uses the power of LLMs to do anything with your input to make any type of output.

![image](https://github.com/lks-ai/anynode/assets/163685473/43043c8f-24f6-4693-bc9e-43666cda78b3)

![image](https://github.com/lks-ai/anynode/assets/163685473/a0596d98-911e-4a93-b0f7-6f6a8782d49d)

[![Watch the video](https://img.youtube.com/vi/f52K5pkbZy8/maxresdefault.jpg)](https://youtu.be/f52K5pkbZy8)

## Install

1. Clone this repository into `comfy/custom_nodes` (sorry, not on Manager just yet!)
2. Make sure you have the `openai` module installed through pip: `pip install openai`
3. Add your `OPENAI_API_KEY` variable to your Environment Variables. [How to get your OpenAI API key](https://platform.openai.com/docs/quickstart)
4. Restart Comfy
5. In ComfyUI double-click and search for `AnyNode` or you can find it in Nodes > utils

## How it Works

1. Put in what you want the node to do with the input and output.
2. Connect it up to anything on both sides
3. Hit `Queue Prompt` in ComfyUI

AnyNode codes a python function based on your request and whatever input you connect to it to generate the output you requested which you can then connect to compatible nodes.

![image](https://github.com/lks-ai/anynode/assets/163685473/1245aa94-fa4d-4490-a3f4-5e8b9918ca28)

Warning: Because of the ability to link ANY node, you have to make sure it nails the output. 

## Caveats
- I have no idea how far you can take this nor it's limits
- LLMs can't read your mind. To make complex stuff in one node you'd have to know a bit about programming
- Right now you can only see code the LLM generates in the console
- Currently there is no way for it to "edit" the code but that is coming soon
- You currently need an OpenAI API Key
- Can't make a sandwich

## Strengths
- You can use as many of these as you want in your workflow creating possibly complex node groups
- Really great at single purpose nodes
- Uses OpenAI API for simple access to the latest and greatest in generation models
- Technically you could point this at vLLM or Ollama for you LocalLLM fans
- Can use most of the popular python libraries and most standard like (numpy, torch, collections, re)
- Ability to make more complex nodes that use inputs like MODEL, VAE and CLIP with input type awareness
- Error Mitigation: Auto-correct errors it made in code (just press `Queue Prompt` again)
- Incremental Code editing (the last generated function serves as example for next generation)

## Coming Soon
- Saving the generated functions in your workflows
- Multiple Inputs and outputs
- Use Local LLMs like vLLM and Ollama (any that support openai chat completions)

## If you're still here
Let's enjoy some stuff I made while up all night!

![image](https://github.com/lks-ai/anynode/assets/163685473/02801f5c-9f67-40f1-83a7-a93e6103d362)
This one, well... the prompts explain it all, but TLDR; It takes an image as input and outputs only the red channel of that image.

![Screenshot from 2024-05-26 01-30-40](https://github.com/lks-ai/anynode/assets/163685473/4cfe5b0b-d515-4f9d-9d86-eff1a08595ed)
Here I use three AnyNodes: One to load a file, one to summarize the text in that file, and the other to just do some parsing of that text. No coding needed.

![image](https://github.com/lks-ai/anynode/assets/163685473/4bc5c6c0-ca56-4f4c-88d5-5339b6d5ada1)
I took that Ant example a bit further and added in the normal nodes to do img2img with my color transforms from AnyNode

