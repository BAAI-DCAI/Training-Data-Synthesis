import os
import re
import shutil
import argparse
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from diffusers.models import AutoencoderKL
from diffusers.utils import load_image

from Generation.data.ImageNet1K import create_ImageNetFolder
# Dataset
from data.new_load_data import get_generation_dataset

print("Package Load Check Done")


def load_caption_dict(image_names,caption_path):
    class_lis = set([image_name.split("/")[0] for image_name in image_names])
    dict_lis = []
    for class_name in class_lis:
        with open(os.path.join(caption_path,f"{class_name}.json"), 'r') as file:
            dict_lis.append(json.load(file))
    caption_dict = {key: value for dictionary in dict_lis for key, value in dictionary.items()}
    return caption_dict

def group_lists(list1, list2, list3, list4, list5):
    grouped_data = {}
    for idx, item in enumerate(list1):
        if item not in grouped_data:
            grouped_data[item] = ([list2[idx]], [list3[idx]], [list4[idx]], [list5[idx]])
        else:
            grouped_data[item][0].append(list2[idx])
            grouped_data[item][1].append(list3[idx])
            grouped_data[item][2].append(list4[idx])
            grouped_data[item][3].append(list5[idx])

    grouped_list = [(key, grouped_data[key][0], grouped_data[key][1], grouped_data[key][2], grouped_data[key][3]) for key in grouped_data]
    return grouped_list
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="imagenette", help="Which Dataset")
    parser.add_argument("--index",default=0,type=int,help="split task")
    parser.add_argument("--version",default="v57",type=str,help="out_version")
    parser.add_argument("--lora_path",default=None,type=str,help="lora path")
    parser.add_argument("--batch_size",default=8,type=int,help="batch size")
    parser.add_argument('--use_caption', default='blip2', type=str, help="use caption model")
    parser.add_argument('--img_size',default=512,type=int, help='Generation Image Size')

    parser.add_argument('--method', default='SD_T2I', type=str, help="generation method")
    parser.add_argument('--use_guidance', default='No', type=str, help="guidance token")
    parser.add_argument('--if_SDXL',default='No', type=str, help="SDXL")
    parser.add_argument('--if_full',default='Yes',type=str, help='singleLora')
    parser.add_argument('--if_compile',default='No',type=str, help='compile?')
    parser.add_argument('--image_strength',default=0.75,type=float,help="init image strength")
    parser.add_argument('--nchunks',default=8,type=int,help="No. subprocess")
    
    parser.add_argument("--imagenet_path",default="",type=str,help="path to imagenet")
    parser.add_argument("--syn_path",default="",type=str,help="path to synthetic data")
    
    # Parameters
    parser.add_argument('--cross_attention_scale', default=0.5, type=float, help="lora scale")
    parser.add_argument('--ref_version',default='v120',type=str, help='version to refine')
    
    args = parser.parse_args()
    return args

class StableDiffusionHandler:
    def __init__(self, args):
        self.args = args
        """
        (Pdb) print(self.args)
        Namespace(batch_size=24, cross_attention_scale=0.5, dataset='imagenette', if_SDXL='No', if_compile='No', if_full='Yes', image_strength=0.75, img_size=512, index=0, lora_path='./LoRA/checkpoint/gt_dm_v1', method='SDI2I_LoRA', nchunks=8, ref_version='v120', use_caption='blip2', use_guidance='Yes', version='v1')
        """
        self.method = args.method   # SDI2I_LoRA
        self.if_SDXL = False
        self.use_guidance_tokens = True
        self.if_full = True 
        self.if_compile = False
        
        self.controlnet_scale = 1.0
        self.lora_path = args.lora_path 
        self.inference_step = 30
        self.guidance_scale = 2.0
        self.cross_attention_scale = args.cross_attention_scale  # 0.5
        self.init_image_strength = args.image_strength  # 0.75
        self.scheduler = "UniPC"
        self.img_size = args.img_size   # 512
        
    ### Get Pipelines
    def get_stablediffusion(self, stablediffusion_path, lora=None):
        pipe = StableDiffusionPipeline.from_pretrained(
            stablediffusion_path, safety_checker=None, torch_dtype=torch.float16, add_watermarker=False
        )
        if lora:
            print("Load LoRA:", os.path.join(self.lora_path,lora))
            pipe.unet.load_attn_procs(os.path.join(self.lora_path,lora))

        pipe = self.set_scheduler(pipe)
        pipe.to("cuda")
        if self.if_compile:
            print("Compile UNet")
            torch._dynamo.config.verbose = True
            pipe.unet = torch.compile(pipe.unet)
        pipe.enable_model_cpu_offload()
        return pipe
    
    def get_img2img(self,img2img_path, lora=None):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(img2img_path, safety_checker=None, torch_dtype=torch.float16)
        if lora:
            print("Load LoRA:", os.path.join(self.lora_path,lora))
            pipe.unet.load_attn_procs(os.path.join(self.lora_path,lora))
        pipe = self.set_scheduler(pipe)
        pipe.to("cuda")
        if self.if_compile: # False
            print("Compile UNet")
            torch._dynamo.config.verbose = True
            pipe.unet = torch.compile(pipe.unet)
        pipe.enable_model_cpu_offload()
        
        return pipe
    
    def set_scheduler(self, pipe):
        if self.scheduler == "UniPC":   #! scheduler
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        elif self.scheduler == "DPM++2MKarras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True)
        elif self.scheduler == "DPM++2MAKarras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True, algorithm_type="sde-dpmsolver++")
        return pipe

    def get_subdataset_loader(self, real_dst_train, bsz, num_chunks=8):
        # split Task
        chunk_size = len(real_dst_train) // num_chunks
        chunk_index = self.args.index
        if chunk_index == num_chunks-1:
            subset_indices = range(chunk_index*chunk_size, len(real_dst_train))
        else:
            subset_indices = range(chunk_index*chunk_size, (chunk_index+1)*chunk_size)
        subset_dataset = Subset(real_dst_train, indices=subset_indices)
        dataloader = DataLoader(subset_dataset, batch_size=bsz, shuffle=False, num_workers=4)
        return dataloader

    ### Generate
    def generate_sd(self,prompts,negative_prompts):
        images = self.pipe(prompts, 
            num_inference_steps=self.inference_step,
            negative_prompt=negative_prompts,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            guidance_scale=self.guidance_scale
            ).images
        
        return images
    
    def generate_sd_lora(self,prompts,negative_prompts, image_names, prev_class_id):
        class_ids = [image_name.split("/")[0] for image_name in image_names]
        groups = group_lists(class_ids, prompts, negative_prompts, negative_prompts, image_names)
        print("Group:",len(groups))
        images = []
        for group in groups:
            class_id, prompts, negative_prompts, _, img_names = group
            if not class_id == prev_class_id and not self.if_full:
                self.pipe = self.get_stablediffusion(class_id)
            if self.use_guidance_tokens:
                guidance_tokens = self.get_guidance_tokens_v2(class_id, img_names)
            else:
                guidance_tokens = None

            sub_images = self.pipe(prompts,
                num_inference_steps=self.inference_step,
                negative_prompt=negative_prompts,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                guidance_scale=self.guidance_scale,
                cross_attention_kwargs={"scale": self.cross_attention_scale},
                guidance_tokens = guidance_tokens
                ).images
            images.extend(sub_images)
        return images, class_id

    def generate_img2img(self,prompts,init_images,negative_prompts):
        images = self.pipe(prompts,
            num_inference_steps=self.inference_step,
            image=init_images,
            negative_prompt=negative_prompts,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            guidance_scale=self.guidance_scale,
            strength=self.init_image_strength
            ).images
        
        return images
    
    def generate_img2img_lora(self,prompts,init_images,negative_prompts, image_names, prev_class_id, class_names=None):
        if self.args.dataset in ['imagenette','imagenet100','imagenet1k']:
            class_ids = [image_name.split("/")[0] for image_name in image_names]
        groups = group_lists(class_ids, prompts, init_images, negative_prompts, image_names)
        print("Group:",len(groups))
        images = []
        for group in groups:
            class_id, prompts, init_images, negative_prompts, img_names = group
            if not class_id == prev_class_id and not self.if_full:
                self.pipe = self.get_img2img(class_id)
            if self.use_guidance_tokens:
                guidance_tokens = self.get_guidance_tokens_v2(class_id, img_names)
            else:
                guidance_tokens = None
            
            sub_images = self.pipe(prompts,
                num_inference_steps=self.inference_step,
                image=init_images,
                negative_prompt=negative_prompts,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                guidance_scale=self.guidance_scale,
                strength=self.init_image_strength,
                cross_attention_kwargs={"scale": self.cross_attention_scale},
                guidance_tokens = guidance_tokens
                ).images
            images.extend(sub_images)
        
        return images, class_id
    
    ### Misc
    def get_pipe(self,pid):
        if self.method in ['SDI2I_LoRA']:
            pipe = self.get_img2img(pid)
        elif self.method in ['SDT2I_LoRA']: # SDI2I_LoRA'
            pipe = self.get_stablediffusion(pid)
        elif self.method in ['SDI2I']:
            pipe = self.get_img2img()
        elif self.method in ['SDT2I']:
            pipe = self.get_stablediffusion()
            
        return pipe
    
    def get_misc(self):
        img_size = (self.img_size, self.img_size)   # (512, 512)
        bsz = self.args.batch_size  # 24
        out_version = self.args.version # "v52"
        create_ImageNetFolder(root_dir=f'{self.args.imagnet_path}/train', out_dir=f"{self.args.syn_path}/train")
        ImageNetPath = self.args.imagnet_path
        dataset = self.args.dataset # "imagenette"
        use_caption = True if self.args.use_caption == 'blip2' else False
        caption_path = "./ImageNet_BLIP2_caption_json/ImageNet_BLIP2_caption_json"
        ###
        print('Image Size',img_size)
        print('Batch Size',bsz)
        print("Use BLIP Caption", use_caption)
        ###
        
        return img_size, bsz, out_version, use_caption, caption_path, ImageNetPath, dataset
    
    def generate(self, prompts,init_images,negative_prompts, image_names, prev_class_id,class_names=None):
        # Generate
        if self.method in ['SDI2I_LoRA']:
            images, prev_class_id = self.generate_img2img_lora(prompts,init_images,negative_prompts, image_names, prev_class_id,class_names)
        elif self.method in ['SDT2I_LoRA']:
            images, prev_class_id = self.generate_sd_lora(prompts, negative_prompts, image_names, prev_class_id)    
        elif self.method in ['SDI2I']:
            images = self.generate_img2img(prompts,init_images,negative_prompts)
        elif self.method in ['SDT2I']:
            images = self.generate_sd(prompts,negative_prompts)
        elif self.method in ['SDXLRefine']:
            images = self.generate_sdxl_refine(prompts,init_images,negative_prompts)
            
        return images, prev_class_id
    
    def get_prompt(self, use_caption, image_names,class_names,caption_path,bs):
        if use_caption:
            base_prompts = [f"photo of {c}" for c in class_names]
            caption_dict = load_caption_dict(image_names, caption_path)
            caption_suffix = [caption_dict[f"{image_name.split('/')[-1]}.JPEG"] for image_name in image_names]
            prompts = [f"{base_prompts[n]}, {caption_suffix[n]}, best quality" for n in range(bs)]
        else:
            prompts = [f"{c}, photo, best quality" for c in class_names]
            
        return prompts
    
    def get_guidance_tokens_v2(self, class_id, image_names):
        if self.args.dataset in ['imagenette','imagenet100','imagenet1k']:
            root='./LoRA/CLIPEmbedding/train'
            

        dir_path = os.path.join(f"{root}", class_id)
        if self.args.dataset in ['imagenette','imagenet100','imagenet1k']:
            # sampled_files = [os.path.join(dir_path,f"{img_name.split('/')[-1]}.pt") for img_name in image_names]
            sampled_files = [f"{img_name.split('/')[-1]}.pt" for img_name in image_names]
            
        feature_dist_samples = [torch.load(os.path.join(dir_path, f)) for f in sampled_files]
        guidance_tokens = torch.stack(feature_dist_samples, dim=1).view(len(image_names),1,-1)
            
        return guidance_tokens
        
    ### Dataset Generation Pipe
    def generate_ImageNet1k(self):
        print(f"Generation: {self.method}")
        print("Full",self.if_full)
        
        prev_class_id = 'all' if self.if_full else None
        self.pipe = self.get_pipe(prev_class_id) # all, sdi2i_lora   
        img_size, bsz, out_version, use_caption, caption_path, ImageNetPath, dataset = self.get_misc()     
        real_dst_train = get_generation_dataset(ImageNetPath, split="train",subset=dataset,filelist="file_list.txt")
        #! just a subset, 8 gpu for running. 
        dataloader = self.get_subdataset_loader(real_dst_train, bsz, num_chunks=self.args.nchunks)
        
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            targets, image_paths, image_names, class_names = batch
            bs = len(image_paths)
            out_paths = [os.path.join(f"{self.args.syn_path}/train",f'{image_names[idx]}.jpg') for idx in range(bs)]
            if os.path.exists(out_paths[-1]):
                continue
            
            prompts = self.get_prompt(use_caption, image_names,class_names,caption_path,bs)
                
            negative_prompts = ["distorted, unrealistic, blurry, out of frame, cropped, deformed" for n in range(bs)]
            
            if self.method in ['SDI2I_LoRA', 'SDI2I', 'SDXLRefine']:
                init_images = [Image.open(image_path).convert("RGB").resize(img_size) for image_path in image_paths]
            else:
                init_images = None
            
            images, prev_class_id = self.generate(prompts,init_images,negative_prompts, image_names, prev_class_id)

            # Save Image
            for idx,image in enumerate(images):
                image.save(out_paths[idx])
                
        # Copy Label
        shutil.copy(f"{ImageNetPath}/train/file_list.txt", f"{self.args.syn_path}/train")

    def generate_pipeline(self):
        if self.args.dataset in ['imagenet1k','imagenette','imagenet100']:
            self.generate_ImageNet1k()
            
        
def main():
    args = get_args()
    # import pdb; pdb.set_trace()
    handler = StableDiffusionHandler(args)
    handler.generate_pipeline()
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()