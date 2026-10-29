from .models import MODEL_FACTORIES

def run_benchmark():

  for model_name, model_factory in MODEL_FACTORIES.items():
    print(f"TODO => {model_name}")
    model = model_factory()
    # print(model)

