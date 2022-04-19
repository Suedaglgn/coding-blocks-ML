import json
import os
import dill as dill


def save_model(pipeline, args, train_acc, test_acc):
    os.makedirs(os.path.join(args.output, args.name))
    with open(os.path.join(args.output, args.name + "/model.pkl"), "wb") as f:
        dill.dump(pipeline, f)

    with open(os.path.join(args.output, args.name + "/result.json"), "w") as f:
        f.write(json.dumps({"train_accuracy": train_acc, "test_accuracy": test_acc}))


def load_model(model_path):
    """
    Loads ML model from file
    :param model_path: path to model folder (path contains only folder name not model name with .pkl extension)
    :return:
    """
    with open(model_path + "/model.pkl", "rb") as f:
        model = dill.load(f)
    f.close()
    return model
