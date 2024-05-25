# AnyNode v0.1 (🍄 beta)

A Node that uses the power of LLMs to do anything with your input to make any type of output.

![image](https://github.com/lks-ai/anynode/assets/163685473/43043c8f-24f6-4693-bc9e-43666cda78b3)

## How it Works

AnyNode codes a python function based on your request and whatever input you connect to it to generate the output you requested which you can then connect to compatible nodes.

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
