from tqdm import tqdm
import wandb

def construct_label_metrix(labels):
    matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    return matrix

def special_epoch_for_loading_speed_test(activate_wandb, total_epochs, epoch, dataloader, model, optimizer, criterion, device):
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    epoch_loss = 0.0
    for step, (image_input_batch, dna_input_batch, label_for_train_batch) in pbar:
        optimizer.zero_grad()
        image_input_batch = image_input_batch.to(device)
        dna_input_batch = dna_input_batch.to(device)
        label_for_train_batch = label_for_train_batch.to(device)
        label_for_train_batch = construct_label_metrix(label_for_train_batch).to(device)

        _ = image_input_batch.size()
        _ = dna_input_batch.size()

        # image_input_output, dna_input_output = model(image_input_batch, dna_input_batch)
        #
        # loss = criterion(image_input_output, dna_input_output, label_for_train_batch)
        #
        # epoch_loss = epoch_loss + loss.item()