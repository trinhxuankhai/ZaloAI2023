import time
import googletrans
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

translator = googletrans.Translator()
model_name_or_path = "TheBloke/Llama-2-13B-GGML"
model_basename = "llama-2-13b.ggmlv3.q4_1.bin"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
lcpp_llm = None
lcpp_llm = Llama(
  model_path=model_path,
  n_threads=2, # CPU cores
  n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
  n_gpu_layers=32 # Change this value based on your model and your GPU VRAM pool.
)

caption = "Phụ nữ là phải đẹp Hỗ trợ tăng vòng 1 tự nhiên cho chị em	Viên sủi hỗ trợ tăng vòng 1, tăng cường nội tiết tố	Sản phẩm không phải là thuốc, không thay thế thuốc chữa bệnh"
fewshot_in0 = "Chị em nào ngực nhỏ lâu năm, kém sắc sau sinh Muốn đẹp nở, săn chắc	Để lại SĐT em chỉ cách này	Sản phẩm không phải thuốc, không có tác dụng thay thế thuốc"
fewshot_in1 = "Cải thiện vòng 1 căng tròn, săn chắc quyến rũ dành cho chị em ngoài 30 tuổi. Quan tâm ngay	Vòng 1 căng tròn săn chắc tự tin hơn	Sản phẩm này không phải là thuốc, không có tác dụng thay thế"
fewshot_out0 = "Người phụ nữ chụp ảnh tự sướng trước gương, bức ảnh lấy cảm hứng từ Kim Jeong-hui, instagram, mặc áo bó sát, ngực khủng!!, cắt dán, cắt tròn"
fewshot_out1 = "một vài phụ nữ đứng cạnh nhau, instagram, tachisme, ngực khủng!, phụ nữ trẻ châu Á, 4k. chất lượng cao, làn da trắng sữa"
prompts = f'''Mô tả lại ảnh quảng cáo từ đoạn quảng cáo sau\n\nĐoạn quảng cáo : {fewshot_in0}\nCâu mô tả ảnh quảng cáo: {fewshot_out0}\n\nĐoạn quảng cáo : {fewshot_in1}\nCâu mô tả ảnh quảng cáo: {fewshot_out1}\n\nĐoạn quảng cáo : {caption}\nCâu mô tả ảnh quảng cáo: '''
prompts = translator.translate(prompts, src='vi', dest='en').text

response=lcpp_llm(prompt=prompts, max_tokens=256, temperature=0.9, top_p=0.95,
                  repeat_penalty=1.2, top_k=150,
                  echo=True)
print(response["choices"][0]["text"])
response["choices"][0]["text"][len(prompts):].split('\n\n')[0]