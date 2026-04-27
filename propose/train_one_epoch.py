import time
import torch
import warnings

warnings.filterwarnings("ignore")

def print_summary(epoch, i, nb_batch, loss_dict, batch_time,
                  average_loss_dict, average_time, mode, lr, logger=None):
    '''
        mode = 'Train' or 'Test'
        loss_dict = dict of current losses e.g. {'total_loss': 1.2, 'metric_loss': 0.8, 'ce_loss': 0.4}
        average_loss_dict = dict of average losses
    '''
    summary = f'    [{mode}] Epoch: [{epoch}][{i}/{nb_batch}]    '
    string = ''
    
    for name, val in loss_dict.items():
        avg_val = average_loss_dict.get(name, val)
        string += f'{name}: {val:.4f} (Avg {avg_val:.4f})   '
        
    if mode == 'Train' and lr is not None:
        string += f'LR: {lr:.2e}   '
        
    string += f'BatchTime: {batch_time:.3f}s (Avg {average_time:.3f}s)'
    
    summary += string
    
    if logger:
        logger.info(summary)
    else:
        print(summary)


def train_one_epoch(data_loader, model, criterion, optimizer, writer, epoch, lr_scheduler, logger, device, cfg):
    """
    Train the model for one epoch.
    """
    model.train()
    
    logging_mode = 'Train'
    end = time.time()
    
    time_sum = 0
    loss_sum = 0
    
    # Track accumulated losses for average computation
    accumulated_losses = {}
    
    for i, batch in enumerate(data_loader, 1):
        images = batch['img'].to(device)
        labels = batch['v_id']
        
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)

        preds = model(images)
        
        # Compute loss
        # The criterion handles the output dict from the Baseline model 
        # (e.g. {'metric_feat': ..., 'bn_feat': ..., 'logits': ...}) and the labels.
        out_loss = criterion(preds, labels)
        
        # Parse loss dictionary if criterion returns multiple losses
        if isinstance(out_loss, tuple): 
            if len(out_loss) == 3: # (total_loss, metric_loss, ce_loss)
                total_loss = out_loss[0]
                loss_dict = {'total_loss': total_loss.item(), 'metric_loss': out_loss[1].item(), 'ce_loss': out_loss[2].item()}
            elif len(out_loss) == 2: # (total_loss, ce_loss)
                total_loss = out_loss[0]
                loss_dict = {'total_loss': total_loss.item(), 'ce_loss': out_loss[1].item()}
        elif isinstance(out_loss, dict):
            total_loss = out_loss.get('total_loss', list(out_loss.values())[0])
            loss_dict = {k: v.item() for k, v in out_loss.items()}
        else:
            total_loss = out_loss
            loss_dict = {'total_loss': total_loss.item()}

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Time calculation
        batch_time = time.time() - end
        end = time.time()

        time_sum += batch_time
        loss_sum += total_loss.item()
        
        for key, value in loss_dict.items():
            accumulated_losses[key] = accumulated_losses.get(key, 0) + value
            
        average_time = time_sum / i
        average_loss = loss_sum / i
        average_loss_dict = {key: value / i for key, value in accumulated_losses.items()}
        
        print_every = cfg.training.print_every if hasattr(cfg.training, 'print_every') else 10
        
        if i % print_every == 0 or i == len(data_loader):
            lr = min(g["lr"] for g in optimizer.param_groups)
            print_summary(epoch, i, len(data_loader), loss_dict, batch_time,
                          average_loss_dict, average_time, logging_mode, lr, logger)

            if writer:
                step = (epoch - 1) * len(data_loader) + i
                for key, value in loss_dict.items():
                    writer.add_scalar(f'{logging_mode}/{key}', value, step)
                writer.add_scalar(f'{logging_mode}/LR', lr, step)

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss