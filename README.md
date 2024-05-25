# AnyNode v0.1 (🍄 beta)

A ComfyUI Node that uses the power of LLMs to do anything with your input to make any type of output.

![image](https://github.com/lks-ai/anynode/assets/163685473/43043c8f-24f6-4693-bc9e-43666cda78b3)

![image](https://github.com/lks-ai/anynode/assets/163685473/309c0e0d-587b-4dc4-a096-0e4975ba6e76)

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
- LLMs can't read your mind. To make complex stuff in one node you'd have to be great at prompting
- Right now you can only see code the LLM generates in the console
- Currently there is no way for it to "edit" the code but that is coming soon
- You currently need an OpenAI API Key

## Strengths
- You can use as many of these as you want in your workflow creating possibly complex node groups
- Uses OpenAI API for simple access to the latest and greatest in generation models
- Technically you could point this at vLLM or Ollama for you LocalLLM fans

## Coming Soon
- Multiple Inputs and outputs
- Some sort of type context map for it's coding abilities
- Ability to make more complex nodes that use inputs like MODEL, VAE and CLIP with input type awareness
- A more complete system prompt
- Error Mitigation (You to LLM: Fix it yourself!)
