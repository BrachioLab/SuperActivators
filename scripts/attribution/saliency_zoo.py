import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSMGrad:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=50, alpha=0.001, early_stop=True, use_sign=False, use_softmax=False):
        dt = data.clone().detach().requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            if use_softmax:
                tgt_out = torch.diag(F.softmax(output, dim=-1)[:, target]).unsqueeze(-1)
            else:
                tgt_out = torch.diag(output[:, target]).unsqueeze(-1)
            tgt_out.sum().backward()
            grad = dt.grad.detach()
            
            for i, idx in enumerate(leave_index):
                grads[idx].append(grad[i:i+1].clone())
            
            if use_sign:
                data_grad = dt.grad.detach().sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(total_grad, -self.epsilon/255, self.epsilon/255)
                dt.data = torch.clamp(data + total_grad, self.data_min, self.data_max)
            else:
                data_grad = grad / grad.view(grad.shape[0], -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(adv_data, self.data_min, self.data_max)
                
            for i, idx in enumerate(leave_index):
                hats[idx].append(dt[i:i+1].data.clone())
            
            if early_stop:
                adv_pred = model(dt).argmax(-1)
                removed_index = np.where((adv_pred != target).cpu())[0]
                keep_index = np.where((adv_pred == target).cpu())[0]
                if len(keep_index) == 0: break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]

        dt_final = torch.cat([hat[-1] for hat in hats], dim=0).requires_grad_(True)
        adv_pred = model(dt_final)
        model.zero_grad()
        tgt_out = torch.diag((F.softmax(adv_pred, dim=-1) if use_softmax else adv_pred)[:, target_clone]).unsqueeze(-1)
        tgt_out.sum().backward()
        
        final_grad = dt_final.grad.detach()
        for i in range(final_grad.shape[0]):
            grads[i].append(final_grad[i:i+1].clone())
            
        return dt_final, (adv_pred.argmax(-1) != target_clone), adv_pred, \
               [torch.cat(h, dim=0) for h in hats], [torch.cat(g, dim=0) for g in grads]

class FGSMGradSingle:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=50, alpha=0.001, early_stop=True, use_sign=False, use_softmax=False):
        dt = data.clone().detach().requires_grad_(True)
        hats, grads = [data.clone()], []
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            tgt_out = (F.softmax(output, dim=-1) if use_softmax else output)[:, target]
            grad = torch.autograd.grad(tgt_out, dt)[0]
            grads.append(grad.clone())
            
            if use_sign:
                adv_data = dt + alpha * grad.detach().sign()
                total_grad = torch.clamp(adv_data - data, -self.epsilon/255, self.epsilon/255)
                dt.data = torch.clamp(data + total_grad, self.data_min, self.data_max)
            else:
                dt.data = torch.clamp(dt - alpha * (grad / grad.norm()) * 100, self.data_min, self.data_max)
            hats.append(dt.data.clone())
            
            if early_stop and model(dt).argmax(-1) != target: break

        adv_pred = model(dt)
        tgt_out = (F.softmax(adv_pred, dim=-1) if use_softmax else adv_pred)[:, target]
        grads.append(torch.autograd.grad(tgt_out, dt)[0].clone())
        return dt, adv_pred.argmax(-1) != target, adv_pred, torch.cat(hats, dim=0), torch.cat(grads, dim=0)

class MFABA:
    def __init__(self, model): self.model = model
    def __call__(self, hats, grads):
        t_list = hats[1:] - hats[:-1]
        total_grads = -torch.sum(t_list * grads[:-1], dim=0)
        return total_grads.unsqueeze(0).detach().cpu().numpy()

class MFABACOS:
    def __init__(self, model): self.model = model
    def __call__(self, data, baseline, hats, grads):
        input_np = hats[0].detach().cpu().numpy()
        base_np = baseline.detach().cpu().numpy()
        diff_base = base_np - input_np
        
        t_list = [np.sum((h.detach().cpu().numpy() - input_np) * diff_base) / (np.linalg.norm(diff_base)**2 + 1e-8) for h in hats]
        t_list = np.array(t_list) / (np.max(t_list) + 1e-8)
        
        dt_list = t_list[1:] - t_list[:-1]
        avg_grads = (grads[:-1] + grads[1:]) / 2
        scaled_grads = avg_grads.view(len(dt_list), -1) * torch.tensor(dt_list, device=grads.device).float().view(-1, 1)
        
        total_grads = torch.sum(scaled_grads.view(len(dt_list), *data.shape[1:]), dim=0).unsqueeze(0)
        return (total_grads * (data - baseline)).detach().cpu().numpy()

class MFABANORM:
    def __init__(self, model): self.model = model
    def __call__(self, data, baseline, hats, grads):
        input_np = hats[0].detach().cpu().numpy()
        t_list = [np.linalg.norm(h.detach().cpu().numpy() - input_np) for h in hats]
        t_list = np.array(t_list) / (np.max(t_list) + 1e-8)
        
        dt_list = t_list[1:] - t_list[:-1]
        avg_grads = (grads[:-1] + grads[1:]) / 2
        scaled_grads = avg_grads.view(len(dt_list), -1) * torch.tensor(dt_list, device=grads.device).float().view(-1, 1)
        
        total_grads = torch.sum(scaled_grads.view(len(dt_list), *data.shape[1:]), dim=0).unsqueeze(0)
        return (total_grads * (data - baseline)).detach().cpu().numpy()

def mfaba_smooth(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255, use_sign=True, use_softmax=True):
    mfaba_obj = MFABA(model)
    attack = FGSMGrad(epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    return np.concatenate([mfaba_obj(hats[i], grads[i]) for i in range(len(hats))], axis=0)

def mfaba_sharp(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255, use_sign=False, use_softmax=True):
    mfaba_obj = MFABA(model)
    attack = FGSMGrad(epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    return np.concatenate([mfaba_obj(hats[i], grads[i]) for i in range(len(hats))], axis=0)

def mfaba_cos(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255, use_sign=False, use_softmax=False):
    dt, _, _, hats, grads = FGSMGradSingle(epsilon, data_min, data_max)(model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    return MFABACOS(model)(data, dt, hats, grads)

def mfaba_norm(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255, use_sign=False, use_softmax=False):
    dt, _, _, hats, grads = FGSMGradSingle(epsilon, data_min, data_max)(model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    return MFABANORM(model)(data, dt, hats, grads)