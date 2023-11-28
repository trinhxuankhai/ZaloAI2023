''' use for cuda 11.8 version
# Install vLLM with CUDA 11.8.
# Replace `cp310` with your Python version (e.g., `cp38`, `cp39`, `cp311`).
pip install https://github.com/vllm-project/vllm/releases/download/v0.2.2/vllm-0.2.2+cu118-cp310-cp310-manylinux1_x86_64.whl

# Re-install PyTorch with CUDA 11.8.
pip uninstall torch -y
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118
'''

''' for cuda 12.1
pip install vllm
'''

import torch
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import os
from tqdm import tqdm

class Prompt:
    def __init__(
            self,
            origin_file:str="data/train/info_trans.csv",
            augument_file:str="data/train/train_caption_v3.json"
    ):
        self.sentence_embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")
        with open(augument_file, 'r') as f:
            self.augument_caption = json.load(f)
        self.origin_caption = pd.read_csv(origin_file)
        os.makedirs('prompt_engineer/prompt_tensor', exist_ok=True)
        self.create_tensor()
        self.caption_embeds = torch.load('prompt_engineer/prompt_tensor/prompt_tensor.pt')

    def create_explicit_prompt(
            self,
            input_path:str="data/test/info_trans.csv",
            output_path:str="prompt_engineer/result/explicit_prompt.json",
            batch_size=4,
            sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=77, repitition_penalty=1.2, top_k=150),
    ):
        '''
        the input path must be csv data type have the same format with origin file
        the origin caption containt implicit description for image so this function will use LLM to generate explicit description to generate image by SD model
        '''
        ids = list(pd.read_csv(input_path)['bannerImage'])
        results = {}
        outputs = []
        prompts = self.create_prompt(input_path=input_path)
        for i in range(0, len(prompts), batch_size):
            input_prompts = prompts[i:i+batch_size]
            generate_seqs = self.llm.generate(input_prompts, sampling_params)
            for generate_seq in generate_seqs:
                outputs.append(generate_seq.outputs[0].text)
        for gen_output, image_ids in zip(outputs, ids):
            results[image_ids] = gen_output
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
        return results
    
    def create_prompt(self, input_path:str="data/test/info_trans.csv"):
        input_data = pd.read_csv(input_path)
        prompts = []
        for i in tqdm(range(len(input_data))):
            sample = input_data.iloc[i]
            caption = sample["caption"].strip('.') + '. ' + sample["description"].strip('.') + '.'
            
            caption_embed = self.sentence_embed_model.encode([caption], convert_to_tensor=True)
            similar_scores = torch.nn.functional.cosine_similarity(caption_embed, self.caption_embeds)
            sort_index = torch.argsort(similar_scores, descending=True, dim=-1)[:2]
            sample_1 = self.origin_caption.iloc[int(sort_index[0])]
            sample_2 = self.origin_caption.iloc[int(sort_index[1])]

            fewshot_in0 = self.cut_long_sentence(sample_1["caption"].strip('.') + '. ' + sample_1["description"] + '.')
            fewshot_in1 = self.cut_long_sentence(sample_2["caption"].strip('.') + '. ' + sample_2["description"] + '.')
            fewshot_out0 = self.cut_long_sentence(self.augument_caption[self.origin_caption.iloc[int(sort_index[0])]["bannerImage"]])
            fewshot_out1 = self.cut_long_sentence(self.augument_caption[self.origin_caption.iloc[int(sort_index[1])]["bannerImage"]])
            prompt = f"Describe the advertisement image from the following advertisement sentence\n\nAdvertisement: {fewshot_in0}\nAdvertisement description: {fewshot_out0}\n\nAdvertisement: {fewshot_in1}\nAdvertisement description: {fewshot_out1}\n\nAdvertisement: {caption}\nAdvertisement photo description:"
            prompts.append(prompt)
        return prompts

    def create_tensor(self):
        prompts = []
        for i in range(len(self.origin_caption)):
            sample = self.origin_caption.iloc[i]
            prompt = sample["caption"].strip('.') + '. ' + sample["description"].strip('.') + '.'
            prompts.append(prompt)
        embed_tensor = self.sentence_embed_model.encode(prompts, convert_to_tensor=True)
        torch.save(embed_tensor, 'prompt_engineer/prompt_tensor/prompt_tensor.pt')

    @staticmethod
    def cut_long_sentence(sentence:str, max_length = 400):
        '''
        if the few shot is too long it must be cut to fit to the model
        '''
        if len(sentence.split()) > max_length:
            cut_sentence = ""
            sentences = sentence.split('.')
            for sen in sentences:
                if len(cut_sentence.split()) + len(sen.split()) <= max_length:
                    cut_sentence += ". " + sen
                else:
                    break
            return cut_sentence
        else:
            return sentence

if __name__ == "__main__":
    prompt_eng = Prompt(origin_file="data/train/info_trans.csv", augument_file="data/train/llava_caption.json")
    output = prompt_eng.create_explicit_prompt(input_path="data/test/info_trans.csv", output_path="data/test/llava_prompt.json")