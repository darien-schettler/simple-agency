""" Example Code for using Stable Diffusion with Runhouse
class SDModel(rh.Module):

    def __init__(self, model_id='stabilityai/stable-diffusion-2-base',
                       dtype=torch.float16, revision="fp16", device="cuda"):
        super().__init__()
        self.model_id, self.dtype, self.revision, self.device = model_id, dtype, revision, device

    def remote_init(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id,
                                                            torch_dtype=self.dtype,
                                                            revision=self.revision).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def predict(self, prompt, num_images=1, steps=100, guidance_scale=7.5):
        return self.pipe(prompt, num_images_per_prompt=num_images,
                         num_inference_steps=steps, guidance_scale=guidance_scale).images


if __name__ == "__main__":
    gpu = rh.ondemand_cluster(name='rh-a10x', instance_type='A10G:1')
    model_gpu = SDModel().to(gpu)
    my_prompt = 'A bright inspiring illustration of a girl running near a house, with a farm in the background.'
    images = model_gpu.predict(my_prompt, num_images=4, steps=50)
    [image.show() for image in images]

    model_gpu.save()

    # You can find more techniques for speeding up Stable Diffusion here:
    # https://huggingface.co/docs/diffusers/optimization/fp16
"""
from simple_agency.config.quantization import bnb_4bit_config
import runhouse as rh
import transformers
import os

from transformers import pipeline, TextStreamer, AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from dotenv import load_dotenv, find_dotenv
from torch import cuda, bfloat16


class HFChatModel(rh.Module):
    def __init__(self,
                 model_id="meta-llama/Llama-2-13b-chat-hf",
                 device="cuda",
                 trust_remote_code=True,
                 model_inference_temperature=0.65,
                 model_inference_max_tokens=4000,
                 model_inference_repetition_penalty=1.0,
                 model_inference_returns_full_text=True,
                 pipe_task="text-generation",
                 streaming=True):
        """ TBD """
        super().__init__()

        # Huggingface model parameters
        self.model_id = model_id
        self.device = device
        self.quantization_config = bnb_4bit_config if '70b' in model_id.lower() else None
        self.trust_remote_code = trust_remote_code

        # Huggingface model inference parameters
        self.model_inference_temperature = model_inference_temperature
        self.model_inference_max_tokens = model_inference_max_tokens
        self.model_inference_repetition_penalty = model_inference_repetition_penalty
        self.model_inference_returns_full_text = model_inference_returns_full_text

        # Huggingface pipeline parameters
        self.pipe_task = pipe_task
        self.streaming = streaming

        # Huggingface pipeline placeholder
        self.hf_pipe = None
        self.pipe_loaded = False

    def remote_init(self):
        """ TBD """

        # Instantiate the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            quantization_config=self.quantization_config,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=bfloat16,
            device_map='auto',
        )

        ### DO I DO THIS??? ###
        model.eval()  # Move the model to the GPU

        # Instantiate the streamer object [optional] and the pipeline [from the tokenizer and model]
        pipe = transformers.pipeline(
            model=model, tokenizer=tokenizer, task=self.pipe_task,
            streamer=TextStreamer(tokenizer, skip_prompt=True) if self.streaming else None,
            return_full_text=self.model_inference_returns_full_text,
            temperature=self.model_inference_temperature,
            max_new_tokens=self.model_inference_max_tokens,
            repetition_penalty=self.model_inference_repetition_penalty,
        )

        # Set the pipeline as an attribute of the module for later use
        self.pipe_loaded = True
        self.hf_pipe = HuggingFacePipeline(pipeline=pipe)

    def predict(self, prompt):
        """ TBD """
        return self.hf_pipe(prompt)


if __name__ == "__main__":
    variant = '13b-chat'
    load_dotenv(find_dotenv())
    basic_config = dict(
        model_id=f"meta-llama/Llama-2-{variant}-hf",
        device="cuda",

        trust_remote_code=True,
        model_inference_temperature=0.7,
        model_inference_max_tokens=4000,
        model_inference_repetition_penalty=1.0,
        model_inference_returns_full_text=True,
        pipe_task="text-generation",
        streaming=True
    )
    # gpu = rh.cluster(
    #     name="a100-cluster",
    #     host=[os.getenv("PAPERSPACE_IP_ADDRESS")],
    #     ssh_creds={
    #         'ssh_user': 'paperspace',
    #         'ssh_private_key': '~/.ssh/id_rsa'
    #     }
    # ).save()

    gpu = rh.cluster("a100-cluster")
    # --> VLLM (IMPROVE LATER FOR PERFORMANCE)
    # gpu.restart_server(restart_ray=True, resync_rh=False)

    if not f"hf-{variant}-model" in gpu.list_keys():
        gpu.delete_keys("all")
        remote_hf_chat_model = HFChatModel(**basic_config).get_or_to("a100-cluster", name=f"hf-{variant}-model")
    else:
        remote_hf_chat_model = gpu.get(f"hf-{variant}-model", remote=True)
    # remote_hf_chat_model.system = gpu

    test_prompt = "what are facts about Australia?"
    test_output_key = remote_hf_chat_model.predict.run(test_prompt)

    test_prompt = "what are facts about Canada?"
    test_output = remote_hf_chat_model.predict(test_prompt)

    print("\n\n... Test Output Key ...\n")
    print(gpu.get(test_output_key))

    print("\n\n... Test Output ...\n")
    print(test_output)
    print("\n\n")

