import importlib

def create_dataset(opt, phase="train"):

    dataset = find_dataset_by_name(opt.DATASET.name)

    # instance 
    dataset = dataset(opt, phase)
    print(f"dataset {type(dataset).__name__} was created!")
    
    return dataset
 

def find_dataset_by_name(dataset_name):
    """
    Given the option --model [dataset_name], the file "src/dataset/dataset_name.py" will be imported
    """
    path_dataset = f"src.datasets.{dataset_name}"
    # path_dataset = f"src.datasets.{dataset_name.lower()}"
    modellib = importlib.import_module(path_dataset)

    # in the file, the calss called DATASET() will be instantiated.
    # it has to be a subclass of Dataset, and it is case-insensitive
    dataset = None
    target_name =  dataset_name.upper() + "Dataset"
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_name.lower():
            dataset = cls

    if dataset is None:
        print(f"In {dataset_name}.py, there should be a class with name {target_name} in lowercase")
        exit(0)

    return dataset