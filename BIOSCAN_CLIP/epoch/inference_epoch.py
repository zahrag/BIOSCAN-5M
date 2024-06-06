from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from epoch.eval_epoch import convert_label_dict_to_list_of_dict, remove_specific_species
import torch
import faiss


def get_feature_and_label(dataloader, model, device, type_of_feature="dna", multi_gpu=False):
    model.eval()
    if type_of_feature not in ['dna', 'image', 'text']:
        raise TypeError(f"{type_of_feature} is not a valid input type")

    if type_of_feature == 'image' and model.image_encoder is None:
        return None, None, None
    if type_of_feature == 'dna' and model.dna_encoder is None:
        return None, None, None
    if type_of_feature == 'text' and model.language_encoder is None:
        return None, None, None

    encoded_feature_list = []
    label_list = []
    file_name_list =[]
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for step, batch in pbar:
            pbar.set_description(f"Getting {type_of_feature} features")
            
            file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
            language_input = {'input_ids': input_ids.to(device), 'token_type_ids': token_type_ids.to(device),
                              'attention_mask': attention_mask.to(device)}

            if type_of_feature == 'dna':
                dna_batch = dna_batch.to(device)
                if multi_gpu:
                    encoded_dna_feature_batch = model.module.dna_encoder(dna_batch)
                else:
                    if model.dna_encoder is None:
                        encoded_dna_feature_batch = F.normalize(dna_batch, dim=-1)
                    else:
                        encoded_dna_feature_batch = F.normalize(model.dna_encoder(dna_batch), dim=-1)

                encoded_feature_list = encoded_feature_list + encoded_dna_feature_batch.cpu().tolist()

            elif type_of_feature == 'image':
                image_input_batch = image_input_batch.to(device)
                if multi_gpu:
                    encoded_image_feature_batch = model.module.image_encoder(image_input_batch)
                else:
                    if model.image_encoder is None:
                        encoded_image_feature_batch = F.normalize(image_input_batch, dim=-1)
                    else:
                        encoded_image_feature_batch = F.normalize(model.image_encoder(image_input_batch), dim=-1)

                encoded_feature_list = encoded_feature_list + encoded_image_feature_batch.cpu().tolist()

            elif type_of_feature == 'text':
                encoded_language_feature_batch = F.normalize(model.language_encoder(language_input), dim=-1)
                encoded_feature_list = encoded_feature_list + encoded_language_feature_batch.cpu().tolist()


            label_list = label_list + convert_label_dict_to_list_of_dict(label_batch)
            file_name_list = file_name_list + list(file_name_batch)

    encoded_feature = np.array(encoded_feature_list)
    return file_name_list, encoded_feature, label_list