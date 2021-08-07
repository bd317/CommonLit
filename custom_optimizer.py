def create_optimizer(model,adjust_task_specific_lr=False):
    named_parameters = list(model.named_parameters())    
    
    roberta_parameters = named_parameters[:388]    
    attention_parameters = named_parameters[388:392]
    regressor_parameters = named_parameters[392:]
        
    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []

    if adjust_task_specific_lr:
      for layer_num, (name, params) in enumerate(attention_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01
        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": Config.task_specific_lr})
      for layer_num, (name, params) in enumerate(regressor_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01
        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": Config.task_specific_lr})   
    else:
      parameters.append({"params": attention_group})
      parameters.append({"params": regressor_group})
    
    increase_lr_every_k_layer = 1
    lrs = np.linspace(1, 5, 24 // increase_lr_every_k_layer)
    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01
        splitted_name = name.split('.')
        lr = Config.lr
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[3]):
            layer_num = int(splitted_name[3])
            lr = lrs[layer_num // increase_lr_every_k_layer] * Config.lr 

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return AdamW(parameters)
