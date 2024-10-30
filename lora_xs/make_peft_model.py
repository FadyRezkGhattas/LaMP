
import yaml
from load_adapters import load_adapter
from peft import get_peft_model, LoraConfig
from lora_xs.initialization_utils import find_and_initialize

def make_peft_model(opts, original_model):
    config = LoraConfig(
        r=opts.rank,
        target_modules=["q", "v"],
        task_type="SEQ_2_SEQ_LM", # assuming a decoder-only model in this example
        lora_alpha=opts.lora_alpha,
        use_rslora=True
        )
    original_model = get_peft_model(original_model, config)

    with open("lora_xs/reconstruct_config.yaml", 'r') as stream:
        reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    adapter_name = "default"  # assuming a single LoRA adapter per module should be transformed to LoRA-XS
    peft_config_dict = {adapter_name: config}
    reconstr_config['svd']['rank'] = opts.rank
    find_and_initialize(
        original_model, peft_config_dict, adapter_name=adapter_name, reconstr_type='svd',
        writer=None, reconstruct_config=reconstr_config, skip_svd=True
    )
    original_model.print_trainable_parameters()
    original_model = load_adapter(original_model, opts.svd_pth)
    return original_model