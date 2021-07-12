from rasa.train import train_nlu

for module_name in modules:
    module_directory = os.path.join(MODULES_BASE_DIR, module_name)
    config_file = <path to config file>
    nlu_data = <path to NLU training folder or file>

    train_nlu(
        config=config_file,
        nlu_data=nlu_data,
        output=module_directory,
        fixed_model_name=module_name,
    )