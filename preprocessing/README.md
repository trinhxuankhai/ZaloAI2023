# Image captioning
## Solution
- Captioning prompt for LLaVA:
```
Given image is advertising about following information: "{input_caption}". What is happening in the image ?
```

## Installation 
- Cloning code from [LLAVA](https://github.com/haotian-liu/LLaVA) source:
```
git clone https://github.com/haotian-liu/LLaVA.git
``` 
- Copy captioning code:
```
cp llava_caption.py LLaVA/llava_caption.py
cd LLaVA
```
- Installing LLaVA following instruction provided from source.
- Run captioning code:
```
python3 llava_caption.py
```