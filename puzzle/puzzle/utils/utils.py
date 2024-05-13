import torch
import torch.nn.functional as F

def gather_log_probs(logits, labels):
    '''
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)
    '''
    # log_probs = F.log_softmax(logits, dim=-1)   # (batch_size, seq_len, vocab_size)
    # log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)) # (batch_size, seq_len, 1)
    # return log_probs_labels.squeeze(-1) # (batch_size, seq_len)
    log_probs_labels = torch.zeros_like(labels, dtype=torch.float32, device=torch.cuda.current_device())
    for i in range(labels.size(0)):
        _log_probs = F.log_softmax(logits[i], dim=-1)
        _log_probs_labels = _log_probs.gather(dim=-1, index=labels[i].unsqueeze(-1))
        log_probs_labels[i] = _log_probs_labels.squeeze(-1)
    return log_probs_labels
