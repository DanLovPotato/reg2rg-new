import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import monai.transforms as transforms
from monai.data import PersistentDataset
import nibabel as nib
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from functools import partial
import torch.nn.functional as F
import tqdm
import random
import pickle
import hashlib
#---Dan---
from Model.fvlm_vit.preprocess import FVLM_ORGAN_MASK_ID, center_crop_like_fvlm_eval
#---Dan---

REGIONS = [
    'abdomen',
    'bone',
    'breast',
    'esophagus',
    'heart',
    'lung',
    'mediastinum',
    'pleura',
    'thyroid',
    'trachea and bronchie',
]

class RadGenomeDataset_Test(PersistentDataset):
    BLANK_VALUE = (-1024 + 400) / 600

    #---Dan---
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, cache_dir,
                 inferenced_id, max_region_size=10, max_img_size=1, image_num=32,
                 region_num=33, max_seq=2048, resize_dim=500, voc_size=32000,
                 force_num_frames=True, use_fvlm=False, fvlm_processed_root=None):
    #---Dan---
        self.inferenced_id = inferenced_id
        # text_tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_tokenizer,
        )
        
        # NOTE: "additoinal_special_tokens" is a type of special token used in tokenizer
        special_token = {
            "additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}
        # NOTE: 'max_img_size' is the max number of images in a single input,
        # 'image_num' is the max number of image tokens in a single image
        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image"+str(i*image_num+j)+">")
            self.image_padding_tokens.append(image_padding_token)
        
        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region"+str(i*region_num+j)+">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region"+str(i*region_num+j)+">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        # treat the token with ID 0 as the pad token
        self.text_tokenizer.pad_token_id = 0
        # treat the token with ID 1 as the bos token
        self.text_tokenizer.bos_token_id = 1
        # treat the token with ID 2 as the eos token
        self.text_tokenizer.eos_token_id = 2
        
        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        #---Dan---
        self.use_fvlm = use_fvlm
        # Keep fVLM crop inputs on the canonical 9-label processed-mask convention
        # used by fvlm/finetune.py and fvlm/eval_finetune.py.  This is deliberately
        # independent of Reg2RG's raw CT / per-region-mask data_folder.
        self.fvlm_processed_root = fvlm_processed_root
        #---Dan---

        #---Dan---
        if self.use_fvlm:
            # Do not infer this from data_folder: smoke datasets can contain a
            # different 10-label merged-mask encoding (lung/pleura gets overwritten).
            if not self.fvlm_processed_root:
                raise ValueError(
                    "fvlm_processed_root is required when use_fvlm=True; it must "
                    "point to the canonical fVLM processed-data root with the "
                    "9-label mask convention."
                )
            split = "train" if os.path.basename(data_folder).startswith("train") else "valid"
            # Canonical root supplies the fVLM image and merged mask consumed below.
            self.fvlm_image_folder = os.path.join(self.fvlm_processed_root, f"processed_{split}_images")
            self.fvlm_mask_folder = os.path.join(self.fvlm_processed_root, f"processed_{split}_masks")
            if not os.path.isdir(self.fvlm_image_folder) or not os.path.isdir(self.fvlm_mask_folder):
                raise FileNotFoundError("fVLM processed image/mask folders are required for exact eval parity")
            self.fvlm_loader = transforms.Compose([
                transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
            ])
        #---Dan---

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64) #NOTE: the target input size of the image
        def threshold(x):
            # threshold at 1
            return x > -1000
        self.region_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            # transforms.ResizeWithPadOrCrop(spatial_size=self.target_size),
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform = self.region_transform, image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))
        
        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

                    if accession_number not in self.accession_to_sentences:
                        continue
                        
                    single_sample = {}
                    volume_name = accession_number.split(".")[0]
                    mask_path = os.path.join(self.mask_folder, 'seg_'+volume_name)
                    single_sample['accnum'] = accession_number
                    # add nii_file to single_sample
                    single_sample['image'] = nii_file

                    for region in REGIONS:
                        mask_file = os.path.join(mask_path, region + '.nii.gz')
                        # NOTE: if the mask file does not exist, skip this sample
                        if not os.path.exists(mask_file):
                            continue
                        
                        # NOTE: if the region is not in the report, set the region report to ''
                        if region in self.accession_to_sentences[accession_number]:
                            region_report = self.accession_to_sentences[accession_number][region]
                        else:
                            region_report = ''
                        
                        single_sample[region] = [mask_file, region_report]

                    samples.append(single_sample)
                    self.paths.append(nii_file)
                    if single_sample['accnum'] in self.inferenced_id:
                        print('Remove: ', single_sample['accnum'])
                        samples.pop()
        
        print('Number of samples: ', len(samples))

        return samples

    def __len__(self):
        return len(self.samples)

    def mask_nii_img_to_tensor(self, img_path, mask_paths, region_transform, image_transform):
        image_nii = nib.load(img_path, mmap=True)
        img_data = np.asarray(image_nii.dataobj)
        img_data = torch.from_numpy(img_data).float()
        #---Dan---
        if self.use_fvlm:
            file_name = os.path.basename(img_path)
            fvlm_data = self.fvlm_loader({
                "image": os.path.join(self.fvlm_image_folder, file_name),
                "label": os.path.join(self.fvlm_mask_folder, file_name),
            })
            fvlm_image = fvlm_data["image"].as_tensor()
            fvlm_mask = fvlm_data["label"].as_tensor()
        #---Dan---

        mask_img_tensors = {}
        fvlm_mask_tensors = {}
        flag = False
        masks = []
        mask_keys = []
        for key, mask_path in mask_paths.items():
            mask_data = nib.load(mask_path, mmap=True)
            mask_data = np.asarray(mask_data.dataobj)
            mask_data = torch.from_numpy(mask_data).float()
            masks.append(mask_data)
            mask_keys.append(key)

            # NOTE: check whether the mask is empty
            if torch.sum(mask_data) == 0:
                continue

            mask_img = img_data * mask_data

            mask_img[mask_data == 0] = -1024
            mask_img = mask_img.unsqueeze(0)

            tensor = region_transform(mask_img)

            hu_min, hu_max = -1000, 200 #NOTE: can directly clip to this range, do not need clip to [-1000, 1000] first
            tensor = torch.clamp(tensor, hu_min, hu_max)

            tensor = (((tensor+400 ) / 600)).float()
            
            # repeat the tensor to have 3 channels
            tensor = tensor.repeat(3, 1, 1, 1)

            tensor = tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)

            #---Dan---
            if self.use_fvlm:
                crop, crop_mask = center_crop_like_fvlm_eval(
                    fvlm_image, fvlm_mask.eq(FVLM_ORGAN_MASK_ID[key]),
                    return_mask=True, mask_value=FVLM_ORGAN_MASK_ID[key],
                )
                mask_img_tensors[key] = crop.unsqueeze(0)
                fvlm_mask_tensors[key] = crop_mask.unsqueeze(0)
            else:
                mask_img_tensors[key] = tensor
            #---Dan---
            flag = True
        if not flag:
            print('No mask: ', img_path)
            import sys
            sys.exit()
        
        # process img_data
        img_data = img_data.unsqueeze(0)
        masks_data = torch.stack(masks, dim=0)
        tensors = image_transform({'img': img_data, 'seg': masks_data})

        img_tensor = tensors['img']
        img_tensor = torch.clamp(img_tensor, hu_min, hu_max)
        img_tensor = (((img_tensor+400 ) / 600)).float()
        # repeat the tensor to have 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1, 1)
        img_tensor = img_tensor.unsqueeze(0) # shape: (1, 3, 256, 256, 64)
        mask_img_tensors['image'] = img_tensor

        masks_tensor = tensors['seg']
        mask_tensors = {}
        for i, key in enumerate(mask_keys):
            mask_tensors[key] = masks_tensor[i].unsqueeze(0)

        return mask_img_tensors, mask_tensors, fvlm_mask_tensors

    def text_add_image_tokens(self, text):
        
        text = '<image>' + self.image_padding_tokens[0] + '</image>' + '. ' + text
        text = "The global information is provided as the context: " + text

        return text

    def text_add_region_tokens(self, text, num_regions):
        region_text = ""
        for i in range(num_regions):
            region_text = region_text + "The region " + str(i) + " is " + '<region>' + self.region_padding_tokens[i] + '</region>. ' 
        text = region_text + text

        return text

    def _build_branch(self, region_order, region_reports, mask_img_tensors,
                      mask_tensors, fvlm_mask_tensors, dropped_regions, sample_id):
        """Build one inference branch using the same inputs as training."""
        is_masked_branch = bool(dropped_regions)
        global_image = mask_img_tensors['image'].clone()
        for region in dropped_regions:
            region_mask = mask_tensors[region][0] > 0.5
            global_image[:, :, region_mask] = self.BLANK_VALUE

        vision_x = {'image': global_image}
        branch_mask_x = {}
        fvlm_mask_x = {}
        region2area = {index: area for index, area in enumerate(region_order)}
        if not is_masked_branch:
            for area in region_order:
                vision_x[area] = mask_img_tensors[area]
                branch_mask_x[area] = mask_tensors[area]
                fvlm_mask_x[area] = fvlm_mask_tensors[area]

        instruction = ("Given the provided global and regional information from this CT scan, please generate a "
                       "comprehensive medical report for each region. First, identify the anatomical area "
                       "corresponding to each region, then provide detailed information about these anatomical "
                       "structures and any abnormalities that are essential. You can refer to the global "
                       "information as the context and take it as a supplement when generating each region report.")
        prompt = self.text_add_region_tokens(instruction, num_regions=len(region2area))
        prompt = self.text_add_image_tokens(prompt)

        combined_report = ""
        for index, area in region2area.items():
            region_report = ("Not evaluable due to missing data."
                             if area in dropped_regions else region_reports[area])
            combined_report += (
                f"The region {index} is {area}: {region_report} "
            )

        text_input = self.text_tokenizer(
            prompt, max_length=self.max_seq, truncation=True, return_tensors="pt"
        )["input_ids"][0]
        return {
            'sample_id': sample_id,
            'lang_x': text_input,
            'vision_x': vision_x,
            'mask_x': branch_mask_x,
            'fvlm_mask_x': fvlm_mask_x,
            'region2area': region2area,
            'dropped_regions': sorted(dropped_regions),
            'is_masked_branch': is_masked_branch,
            'question': prompt,
            'gt_combined_report': combined_report,
        }

    def __getitem__(self, index):
        img_file = self.data[index]['image']

        region_reports = {}
        mask_files = {}
        for key in self.data[index]:
            if key == 'image' or key == 'accnum':
                continue
            mask_file, region_report = self.data[index][key]
            region_reports[key] = region_report
            mask_files[key] = mask_file

        mask_img_tensors, mask_tensors, fvlm_mask_tensors = self.mask_img_to_tensor(img_file, mask_files)

        #NOTE: remove useless regions from region_reports according to mask_img_tensors, only used to compute region prediction accuracy
        for key in list(region_reports.keys()):
            if key not in mask_img_tensors:
                print('Remove region: ', key + ' from ' + img_file)
                region_reports.pop(key)

        region_order = list(region_reports.keys())
        sample_id = os.path.basename(self.data[index]['accnum']).removesuffix('.nii.gz')

        # Make masking stable per accession, including when an interrupted run resumes.
        sample_seed = int.from_bytes(
            hashlib.sha256(sample_id.encode('utf-8')).digest()[:8], 'big'
        )
        sample_rng = random.Random(sample_seed)
        dropped_regions = set()
        if region_order:
            drop_fraction = sample_rng.uniform(0.3, 0.5)
            num_to_drop = max(1, int(len(region_order) * drop_fraction))
            dropped_regions = set(sample_rng.sample(region_order, num_to_drop))

        full_sample = self._build_branch(
            region_order, region_reports, mask_img_tensors, mask_tensors, fvlm_mask_tensors,
            dropped_regions=set(), sample_id=sample_id,
        )
        mask_sample = self._build_branch(
            region_order, region_reports, mask_img_tensors, mask_tensors, fvlm_mask_tensors,
            dropped_regions=dropped_regions, sample_id=sample_id,
        )
        return {
            'acc_num': self.data[index]['accnum'],
            'full': full_sample,
            'mask': mask_sample,
        }
