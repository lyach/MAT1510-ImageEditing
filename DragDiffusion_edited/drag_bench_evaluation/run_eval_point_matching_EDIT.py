# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import argparse
import os
import pickle
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import PILToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from dift_sd import SDFeaturizer
from pytorch_lightning import seed_everything

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting arguments")
    parser.add_argument('--eval_root',
                        action='append',
                        help='root of dragging results for evaluation',
                        required=True)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # using SD-2.1
    dift = SDFeaturizer('stabilityai/stable-diffusion-2-1')

    all_category = [
        'art_work',
        # 'land_scape',
        'building_city_view',
        # 'building_countryside_view',
        # 'animals',
        # 'human_head',
        # 'human_upper_body',
        'human_full_body',
        # 'interior_design',
        # 'other_objects',
    ]

    original_img_root = 'drag_bench_data/'

    for target_root in args.eval_root:
        category_dist = {}  
        category_total_dist = {}  # To accumulate distances per category
        category_total_pairs = {}  # To count pairs per category

        for cat in all_category:
            print(f"\nCategory: {cat}")
            image_dist = []  # To store mean distances for each image in the category
            category_total_dist[cat] = 0.0
            category_total_pairs[cat] = 0

            for file_name in os.listdir(os.path.join(original_img_root, cat)):
                if file_name == '.DS_Store':
                    continue
                with open(os.path.join(original_img_root, cat, file_name, 'meta_data.pkl'), 'rb') as f:
                    meta_data = pickle.load(f)
                prompt = meta_data['prompt']
                points = meta_data['points']

                handle_points = []
                target_points = []
                for idx, point in enumerate(points):
                    cur_point = torch.tensor([point[1], point[0]])
                    if idx % 2 == 0:
                        handle_points.append(cur_point)
                    else:
                        target_points.append(cur_point)

                source_image_path = os.path.join(original_img_root, cat, file_name, 'original_image.png')
                dragged_image_path = os.path.join(target_root, cat, file_name, 'dragged_image.png')

                # To handle missing files
                if not os.path.exists(dragged_image_path):
                    print(f"  Skipping missing file: {dragged_image_path}")
                    continue

                source_image_PIL = Image.open(source_image_path).convert('RGB')
                dragged_image_PIL = Image.open(dragged_image_path).convert('RGB')
                dragged_image_PIL = dragged_image_PIL.resize(source_image_PIL.size, PIL.Image.BILINEAR)

                source_image_tensor = (PILToTensor()(source_image_PIL) / 255.0 - 0.5) * 2
                dragged_image_tensor = (PILToTensor()(dragged_image_PIL) / 255.0 - 0.5) * 2

                _, H, W = source_image_tensor.shape

                ft_source = dift.forward(source_image_tensor,
                                         prompt=prompt,
                                         t=261,
                                         up_ft_index=1,
                                         ensemble_size=2)
                ft_source = F.interpolate(ft_source, (H, W), mode='bilinear')

                ft_dragged = dift.forward(dragged_image_tensor,
                                          prompt=prompt,
                                          t=261,
                                          up_ft_index=1,
                                          ensemble_size=2)
                ft_dragged = F.interpolate(ft_dragged, (H, W), mode='bilinear')

                cos = nn.CosineSimilarity(dim=1)
                all_dist = []  
                for pt_idx in range(len(handle_points)):
                    hp = handle_points[pt_idx]
                    tp = target_points[pt_idx]

                    num_channel = ft_source.size(1)
                    src_vec = ft_source[0, :, hp[0], hp[1]].view(1, num_channel, 1, 1)
                    cos_map = cos(src_vec, ft_dragged).cpu().numpy()[0]  # H, W
                    max_rc = np.unravel_index(cos_map.argmax(), cos_map.shape)  # the matched row,col

                    dist = (tp - torch.tensor(max_rc)).float().norm()
                    all_dist.append(dist.item())

                # Calculate mean distance for the current image
                mean_distance_per_image = np.mean(all_dist) if all_dist else None

                if mean_distance_per_image is not None:
                    image_dist.append(mean_distance_per_image) 
                    category_total_dist[cat] += sum(all_dist)
                    category_total_pairs[cat] += len(all_dist)
                    print(f"  {file_name}: Mean Distance = {mean_distance_per_image:.4f}")
                else:
                    print(f"  {file_name}: No Data")

            # Calculate mean distance for the current category based on all pairs
            if category_total_pairs[cat] > 0:
                category_dist[cat] = category_total_dist[cat] / category_total_pairs[cat]
            else:
                category_dist[cat] = None

        print(f"\nResults for target root: {target_root}")
        for cat in all_category:
            avg_dist = category_dist.get(cat, 'No Data')
            if avg_dist is not None:
                print(f"Category: {cat}")
                print(f"  Avg Distance: {avg_dist:.4f}")
            else:
                print(f"Category: {cat}")
                print(f"  Avg Distance: No Data")


