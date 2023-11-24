import torch
from ctransformers import AutoModelForCausalLM
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
        config = {'max_new_tokens': 77, 'repetition_penalty': 1.2, 'temperature': 0.9, 'stream': False, 'context_length':1024, 'top_k':150, 'top_p':0.95}
        self.sentence_embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.llm = AutoModelForCausalLM.from_pretrained("TheBloke/Amethyst-13B-Mistral-GGUF", model_file="amethyst-13b-mistral.Q4_K_M.gguf", model_type="llama", gpu_layers=50, **config)
        with open(augument_file, 'r') as f:
            self.augument_caption = json.load(f)
        self.origin_caption = pd.read_csv(origin_file)
        if not os.path.exists(r'prompt_engineer/prompt_tensor'):
            os.mkdir('prompt_engineer/prompt_tensor')
            self.create_tensor()
        self.caption_embeds = torch.load('prompt_engineer/prompt_tensor/prompt_tensor.pt')

    def create_explicit_prompt(self, input_path:str="data/test/info_trans.csv", output_path:str="prompt_engineer/result/explicit_prompt.json"):
        '''
        the input path must be csv data type have the same format with origin file
        the origin caption containt implicit description for image so this function will use LLM to generate explicit description to generate image by SD model
        '''
        captions, ids = list(pd.read_csv(input_path)['caption']), list(pd.read_csv(input_path)['bannerImage'])
        outputs = {}
        for caption, cap_id in tqdm(zip(captions, ids)):
            caption_embed = self.sentence_embed_model.encode([caption], convert_to_tensor=True)
            similar_scores = torch.nn.functional.cosine_similarity(caption_embed, self.caption_embeds)[0]
            sort_index = torch.argsort(similar_scores, dim=-1)[::-1][:2]
            fewshot_in0 = self.cut_long_sentence(self.origin_caption["caption"][sort_index[0]])
            fewshot_in1 = self.cut_long_sentence(self.origin_caption["caption"][sort_index[1]])
            fewshot_out0 = self.cut_long_sentence(self.augument_caption[self.origin_caption["bannerImage"][sort_index[0]]])
            fewshot_out1 = self.cut_long_sentence(self.augument_caption[self.origin_caption["bannerImage"][sort_index[1]]])
            prompt = f"Describe the advertisement image from the following advertisement sentence\n\nAdvertisement: {fewshot_in0}\nAdvertisement description: {fewshot_out0}\n\nAdvertisement: {fewshot_in1}\nAdvertisement description: {fewshot_out1}\n\nAdvertisement: {caption}\nAdvertisement photo description:"
            output = self.llm(prompt, stream=False)
            output = output.split('\n')[0].strip()
            outputs[cap_id] = output
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(outputs, f, indent=4)
        return outputs
    
    def generate_ad_object(self, input_path:str="data/test_info.csv", output_path:str="prompt_engineer/result/object.json"):
        '''
        this function will generate what is the advertisement sentence is about
        '''
        outputs = {}
        captions, ids = list(pd.read_csv(input_path)['caption']), list(pd.read_csv(input_path)['bannerImage'])
        for caption, cap_id in tqdm(zip(captions, ids)):
            prompt = f"Đoạn thông tin sau quảng cáo về sản phẩm gì\n\nĐoạn quảng cáo: Đẳng cấp quý ông với dây nịt da cao cấp Crocodile. Cơ hội vàng - sale sập sàn tới 80% Đồ da cao cấp Leather\nSản phẩm: dây nịt\n\nĐoạn quảng cáo: Ưu đãi 60%. Mẫu giày da bò chỉ 399k. Miễn phí ship, xem hàng khi thanh toán\nSản phẩm: giày da\n\nĐoạn quảng cáo: {caption}\nSản phẩm:"
            prompt = self.translator.translate(prompt, src='vi', dest='en').text
            output = self.llm(prompt, stream=False)
            output = output.split('\n')[0].strip()
            outputs[cap_id] = output
        if output_path:
            with open(output_path, "w") as f:
                json.dump(outputs, f, indent=4)
        return outputs

    def create_tensor(self):
        prompts = list(self.origin_caption["caption"])
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
    prompt_eng = Prompt()
    output = prompt_eng.create_explicit_prompt(input_path="data/test/info_trans.csv", output_path="data/test/explicit_prompt.json")
    # output = prompt_eng.generate_ad_object(input_path="path to csv file want to create know object ad is about", output_path="save result path, if set to None it will not save the result")