import torch
import numpy as np

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def select_confident_samples(
    logits,
    top,
    order: bool = False,
):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    selected_num = max(1, int(batch_entropy.size(0) * top))
    idx = torch.argsort(batch_entropy, descending=order)[:selected_num]
    return logits[idx], idx


def compute_energy_metric(zs_outputs, idx, temperature=1.0):
    selected_logits = torch.index_select(zs_outputs, 0, idx)
    energy = -temperature * torch.logsumexp(selected_logits / temperature, dim=1)
    metric_value = -energy.mean().item()
    return metric_value


def compute_mcm_metric(zs_outputs, idx, scale=100.0):
    selected_logits = torch.index_select(zs_outputs, 0, idx)
    probs = torch.softmax(selected_logits / scale, dim=-1)
    return probs.max(dim=-1).values.mean().item()


def compute_random_metric(rng):
    return float(rng.rand())


def compute_cas(zs_outputs,idx):
    all_probs = zs_outputs.softmax(dim=-1)
    selected_probs = torch.index_select(all_probs, 0, idx)
    selected_logits = torch.index_select(zs_outputs, 0, idx)
    
    main_pred = zs_outputs[0, :].argmax(dim=-1)
    selected_preds = selected_logits.argmax(dim=-1)
    consistency = (selected_preds == main_pred).float()
    
    main_prob = all_probs[0, :]
    selected_probs_norm = selected_probs / selected_probs.norm(dim=1, keepdim=True)  # [k, C]
    main_prob_expand = main_prob.unsqueeze(0).expand(len(idx), -1)  # [k, C]
    main_prob_norm = main_prob_expand / main_prob_expand.norm(dim=1, keepdim=True)  # [k, C]
    similarity = (selected_probs_norm * main_prob_norm).sum(dim=1)
    similarity = similarity / similarity.sum(dim=0, keepdim=True)
    cas = (consistency * similarity).sum(dim=0)

    
    return cas.item()

class TPT():
    def __init__(self, model,args, dset=None):
        self.model = model
        self.device = args.gpu
        self.total_samples = 0
        self.metric_rng = np.random.RandomState(0)

    def pre_adaptation(self, args):
        if args.load is not None:
            self.model.prompt_learner.ctx_init_state = args.pretrained_ctx
        with torch.no_grad():
            self.model.reset()
        self.optimizer = torch.optim.AdamW(self.model.prompt_learner.parameters(), args.lr)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=1000)

    def adaptation_process(self, image, images, args):
        assert args.tta_steps > 0

        self.pre_adaptation(args)
        self.total_samples += 1

        with torch.amp.autocast('cuda'):
            output = self.model(images)
        _, selected_idx = select_confident_samples(output, args.selection_p)

        if args.metric == 'cas':
            metric = compute_cas(output, selected_idx)
        elif args.metric == 'energy':
            metric = compute_energy_metric(output, selected_idx)
        elif args.metric == 'mcm':
            metric = compute_mcm_metric(output, selected_idx)
        elif args.metric == 'random':
            metric = compute_random_metric(self.metric_rng)
        else:
            raise ValueError(f"Unknown metric: {args.metric}")
        zs_pred = output[0].argmax().item()

        for step in range(args.tta_steps):
            with torch.amp.autocast('cuda'):
                loss = avg_entropy(output[selected_idx])
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            if step + 1 < args.tta_steps:
                with torch.amp.autocast('cuda'):
                    output = self.model(images)

        tta_outputs = self.model(image)
        tta_pred = tta_outputs[0].argmax().item()

        return_dict = {
            'zs_pred': zs_pred,
            'tta_pred': tta_pred,
            'metric': metric,
        }

        return return_dict


  
  
