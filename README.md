# Navigating the Geometry of Language

**A New Approach to Synthetic Text Generation**

This demo is a practical example of the geometric approach to latent space sampling as described in the paper [Navigating the Geometry of Language: A New Approach to Synthetic Text Generation](https://www.watchful.io/blog/navigating-the-geometry-of-language-a-new-approach-to-synthetic-text-generation). It allows you to generate a dataset of examples for a given prompt using the [OpenAI API](https://platform.openai.com/docs/introduction). You can browse the live demo [here](https://dataset-generator.gpu-demos.watchful.io).

## Quickstart

This demo requires PyTorch to be compiled with CUDA support.

```bash
pip install -r requirements.txt
OPENAI_API_KEY=<KEY> OPENAI_ORGANIZATION=<ORG> streamlit run streamlit_app.py
```
