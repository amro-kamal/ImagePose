
# import copy
# import datetime
# import logging
# import os

# import torch
# from tqdm import tqdm

# from .datasets import ToTensorflow
# from .evaluation import evaluate as e
# from .utils import load_dataset, load_model



# correct, correct5, correct_t_conf, correct_f_conf, wrong_conf, result, result5 =  run_model(model, mydataloader,
#                                                                               list(imagenet_classes.values()), savepath[i], 
#                                                                            pose=pose[i], model_name=model_name, model_zoo=model_zoo, true_classes=true_class[i])



# logger = logging.getLogger(__name__)
# MAX_NUM_MODELS_IN_CACHE = 3


# def device():
#     return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class ModelEvaluator:

#     def _pytorch_evaluator(self, model_name, model, dataset, *args, **kwargs):
#         """
#         Evaluate Model on the given dataset and return the accuracy.
#         Args:
#             model_name:
#             model:
#             dataset:
#             *args:
#             **kwargs:
#         """

#         logging_info = f"Evaluating model {model_name} on dataset {dataset.name} using Pytorch Evaluator"
#         logger.info(logging_info)
#         print(logging_info)
#         for metric in dataset.metrics:
#             metric.reset()
#         with torch.no_grad():
#             result_writer = e.ResultPrinter(model_name=model_name,
#                                             dataset=dataset)

#             for images, target, paths in tqdm(dataset.loader):
#                 images = images.to(device())
#                 logits = model.forward_batch(images)
#                 softmax_output = model.softmax(logits)
#                 if isinstance(target, torch.Tensor):
#                     batch_targets = model.to_numpy(target)
#                 else:
#                     batch_targets = target
#                 predictions = dataset.decision_mapping(softmax_output)
#                 for metric in dataset.metrics:
#                     metric.update(predictions,
#                                   batch_targets,
#                                   paths)
#                 if kwargs["print_predictions"]:
#                     result_writer.print_batch_to_csv(object_response=predictions,
#                                                      batch_targets=batch_targets,
#                                                      paths=paths)

    

#     def _get_datasets(self, dataset_names, *args, **kwargs):
#         dataset_list = []
#         for dataset in dataset_names:
#             dataset = load_dataset(dataset, *args, **kwargs)
#             dataset_list.append(dataset)
#         return dataset_list



#     def _get_evaluator(self, framework):
#         if framework == "tensorflow":
#             return self._tensorflow_evaluator
#         elif framework == 'pytorch':
#             return self._pytorch_evaluator
#         else:
#             raise NameError("Unsupported evaluator")

#     def _remove_model_from_cache(self, framework, model_name):

#         def _format_name(name):
#             return name.lower().replace("-", "_")

#         try:
#             if framework == "pytorch":
#                 cachedir = "/root/.cache/torch/checkpoints/"
#                 downloaded_models = os.listdir(cachedir)
#                 for dm in downloaded_models:
#                     if _format_name(dm).startswith(_format_name(model_name)):
#                         os.remove(os.path.join(cachedir, dm))
#         except:
#             pass

#     def __call__(self, models, dataset_names, *args, **kwargs):
#         """
#         Wrapper call to _evaluate function.

#         Args:
#             models:
#             dataset_names:
#             *args:
#             **kwargs:

#         Returns:

#         """
#         logging.info("Model evaluation.")
#         _datasets = self._get_datasets(dataset_names, *args, **kwargs)
#         for model_name in models:
#             datasets = _datasets
#             model, framework = load_model(model_name, *args)
#             evaluator = self._get_evaluator(framework)
#             if framework == 'tensorflow':
#                 datasets = self._to_tensorflow(datasets)
#             logger.info(f"Loaded model: {model_name}")
#             for dataset in datasets:
#                 # start time
#                 time_a = datetime.datetime.now()
#                 evaluator(model_name, model, dataset, *args, **kwargs)
#                 for metric in dataset.metrics:
#                     logger.info(str(metric))
#                     print(metric)

#                 # end time
#                 time_b = datetime.datetime.now()
#                 c = time_b - time_a

#                 if kwargs["print_predictions"]:
#                     # print performances to csv
#                     for metric in dataset.metrics:
#                         e.print_performance_to_csv(model_name=model_name,
#                                                    dataset_name=dataset.name,
#                                                    performance=metric.value,
#                                                    metric_name=metric.name)
#             if len(models) >= MAX_NUM_MODELS_IN_CACHE:
#                 self._remove_model_from_cache(framework, model_name)

#         logger.info("Finished evaluation.")


import os
from PIL import Image
for name in os.listdir('data/360/YAW/bg1/containership_YAW_360/images_lr600'):
    if name =='.DS_Store':
        continue
    p = name.split('.')[0].split('_')[-1]
    new_name= f'containership_yaw_{p}.png'
    print(new_name)
    img = Image.open(os.path.join('data/360/YAW/bg1/containership_YAW_360/images_lr600', name))
    img.save(os.path.join('data/360/YAW/bg1/containership_YAW_360/new_images_lr600', new_name) )