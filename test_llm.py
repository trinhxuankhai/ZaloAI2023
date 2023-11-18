import googletrans
from ctransformers import AutoModelForCausalLM

translator = googletrans.Translator()

model_id = "TheBloke/Llama-2-13B-GGML"
model_file = 'llama-2-13b.ggmlv3.q4_1.bin'
config = {'max_new_tokens': 77, 'repetition_penalty': 1.2, 'temperature': 0.9, 'stream': False, 'context_length':1024, 'top_k':150, 'top_p':0.95}
llm = AutoModelForCausalLM.from_pretrained(model_id,
                                           model_file=model_file,
                                           model_type="llama",
                                           #lib='avx2', for cpu use
                                           gpu_layers=130, #110 for 7b, 130 for 13b
                                           **config
                                           )

caption = "Ưu đãi 80% chi phí cho 10 bạn răng xấu Trọn bộ bọc răng sứ chỉ 5 triệu"
fewshot_in0 = "Bọc ghế da ô tô Hà Nội với thợ tay nghề trên 7 năm kinh nghiệm, êm ái trên mọi nẻo đường"
fewshot_in1 = "Bạt cách nhiệt phủ ô tô 3 lớp - Chống xước xe, chống nắng nóng bụi bẩn, chống thấm nước."
fewshot_out0 = "Bức ảnh mô tả ghế xe ô tô với những chiếc bọc ghế bằng da"
fewshot_out1 = "Bức ảnh mô tả chiếc bạt phủ trên một chiếc xe ô tô"
prompts = f'''Mô tả lại ảnh quảng cáo từ đoạn quảng cáo sau\n\nĐoạn quảng cáo : {fewshot_in0}\nCâu mô tả ảnh quảng cáo: {fewshot_out0}\n\nĐoạn quảng cáo : {fewshot_in1}\nCâu mô tả ảnh quảng cáo: {fewshot_out1}\n\nĐoạn quảng cáo : {caption}\nCâu mô tả ảnh quảng cáo: '''
prompts = translator.translate(prompts, src='vi', dest='en').text

# 'pipeline' execution
output = llm(prompts, stream=False)

print(output.split('\n\n')[0])